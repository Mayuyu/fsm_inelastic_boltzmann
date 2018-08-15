import numpy as np
from math import pi
from utility import sinc
import pyfftw

class DirectCol3D(object):
    def __init__(self, config, e=None, N=None):
        self._config = config
        self._gamma = config.physical_config.gamma
        if e is None:
            self._e = config.physical_config.e
        else:
            self._e = e

        S = config.domain_config.S
        self._R = 2*S
        self._L = eval(config.domain_config.L)
        if N is None:
            self._N = config.domain_config.N
        else:
            self._N = N

        self._N_R = config.quadrature_config.N_g

        self._Glm = None
        self._l = None
        self._m = None
        
        self._dv = None
        self._v = None
        self._v_norm = None

        self._fftw_plan()

    def _G(self, k, m):
        x, w = np.polynomial.legendre.leggauss(self._N_R)
        r = 0.5*(x + 1)*self._R
        r_w = 0.5*self._R*w
        km = 0.25*(1 + self._e)*k[:,None] - m
        k_m_norm = np.sqrt(km[:,None,None,:,None,None]**2 + km[:,None,None,:,None]**2 + km[:,None,None,:]**2)
        m_norm = np.sqrt(m[:,None,None]**2 + m[:,None]**2 + m**2)
        k_norm = np.sqrt(k[:,None,None]**2 + k[:,None]**2 + k**2)
        return 16*pi**2*np.sum(r_w*r**(self._gamma+2)*(sinc(pi*k_m_norm[...,None]*r/self._L)*sinc(pi*0.25*(1+self._e)*k_norm[...,None,None,None,None]*r/self._L) 
            - sinc(pi*m_norm[...,None]*r/self._L)), axis = (-1))

    def _lm(self):
        l, m = [], []
        for i in range(self._N):
            l.append([])
            m.append([])
        N_half = int(self._N/2)    
        for i in np.arange(-N_half, N_half):
            for j in np.arange(-N_half, N_half):
                if i + j > N_half - 1:
                    l[i+j-N_half].append(i+N_half)
                    m[i+j-N_half].append(j+N_half)
                elif i + j < -N_half:
                    l[i+j+3*N_half].append(i+N_half)
                    m[i+j+3*N_half].append(j+N_half)
                else:
                    l[i+j+N_half].append(i+N_half)
                    m[i+j+N_half].append(j+N_half)

        return l, m

    def save_Glm(self, path):
        N = self._N
        k = np.arange(-N/2, N/2)
        Glm = np.zeros((N, N, N, N, N, N))
        Glm = self._G(k, k)
        l, m = self._lm()
        np.savez(path, Glm=Glm, l=l, m=m)
        
    def get_Glm_from_file(self, path):
        array = np.load(path)
        Glm, l, m = array['Glm'], array['l'], array['m']
        if Glm.shape[0] != self._N:
            raise("The size is not consistent with this configuration. Please choose a different file")
        else:
            self._Glm = Glm
            self._l = l
            self._m = m

    def col(self, f):
        N, l, m = self._N, self._l, self._m
        Q_hat = np.zeros((N, N, N), dtype='complex')
        f_hat = np.fft.fftshift(self._fft3(f))
        for kx in range(N):
            for ky in range(N):
                for kz in range(N):
                    g_hat = self._Glm[kx, ky, kz]*f_hat
                    Q_hat[kx, ky, kz] = np.sum(g_hat[np.ix_(l[kx], l[ky], l[kz])]*f_hat[np.ix_(m[kx], m[ky], m[kz])])
        return np.real(self._ifft3(np.fft.ifftshift(Q_hat)))/N**3

    def _fftw_plan(self):
        array_3d = pyfftw.empty_aligned((self._N, self._N, self._N), dtype='complex128')
        self._fft3 = pyfftw.builders.fftn(array_3d, overwrite_input=True,
                                        planner_effort='FFTW_ESTIMATE', threads=8, avoid_copy=True)
        self._ifft3 = pyfftw.builders.ifftn(array_3d, overwrite_input=True,
                                            planner_effort='FFTW_ESTIMATE', threads=8, avoid_copy=True)


    
