# fast_spec_col_3d

import numpy as np
import pyfftw
from math import pi

EPS = 1e-8
def sinc(x):
    return np.sin(x+EPS)/(x+EPS)

class FastSpectralCollison3D:
    
    def __init__(self, e, gamma, L, N, R, N_R, char='ss005.00012.txt'):
        # legendre quadrature
        x, w = np.polynomial.legendre.leggauss(N_R)
        r = 0.5*(x + 1)*R
        self.r_w = 0.5*R*w
        # spherical points and weight
        sigma = np.loadtxt(char)
        self.s_w = 4*pi/sigma.shape[0]
        # index
        k = np.fft.fftshift(np.arange(-N/2,N/2))
        # dot with index
        rkg = (k[:,None,None,None]*sigma[:,0] 
                    + k[:,None,None]*sigma[:,1] 
                    + k[:,None]*sigma[:,2])[...,None]*r
        # norm of index
        k_norm = np.sqrt(k[:,None,None]**2 + k[:,None]**2 + k**2)
        # gain kernel
        self.F_k_gain = 4*pi*r**(gamma+2)*(np.exp(0.25*1j*pi*(1+e)*rkg/L)
                                           *sinc(0.25*pi*(1+e)*r*k_norm[...,None,None]/L))
        # loss kernel
        self.F_k_loss = 4*pi*r**(gamma+2)
        # exp for fft
        self.exp = np.exp(-1j*pi*rkg/L)
        # sinc
        self.sinc = sinc(pi*r*k_norm[...,None]/L)
        # laplacian
        self.lapl = -pi**2/L**2*k_norm**2
        
        # pyfftw planning of (N, N, N)
        array_3d = pyfftw.empty_aligned((N, N, N), dtype='complex128')
        self.fft3 = pyfftw.builders.fftn(array_3d, overwrite_input=True,
                                   planner_effort='FFTW_ESTIMATE', threads=8, avoid_copy=True)
        self.ifft3 = pyfftw.builders.ifftn(array_3d, overwrite_input=True,
                                     planner_effort='FFTW_ESTIMATE', threads=8, avoid_copy=True)
        # pyfftw planning of (N, N, N, N_R)
        array_4d = pyfftw.empty_aligned((N, N, N, N_R), dtype='complex128')
        self.fft4 = pyfftw.builders.fftn(array_4d, axes=(0,1,2), overwrite_input=True,
                                   planner_effort='FFTW_ESTIMATE', threads=8, avoid_copy=True)
        self.ifft4 = pyfftw.builders.ifftn(array_4d, axes=(0,1,2), overwrite_input=True,
                                     planner_effort='FFTW_ESTIMATE', threads=8, avoid_copy=True)
        # pyfftw planning of (N, N, N, sigma, N_R)
        array_5d = pyfftw.empty_aligned((N, N, N, sigma.shape[0], N_R), dtype='complex128')
        self.fft5 = pyfftw.builders.fftn(array_5d, axes=(0,1,2), overwrite_input=True,
                                   planner_effort='FFTW_ESTIMATE', threads=8, avoid_copy=True)
        self.ifft5 = pyfftw.builders.ifftn(array_5d, axes=(0,1,2), overwrite_input=True,
                                     planner_effort='FFTW_ESTIMATE', threads=8, avoid_copy=True)
        
    def col_full(self, f):
        # fft of f
        f_hat = self.fft3(f)
        # convolution and quadrature
        Q = self.s_w*np.sum(self.r_w*(self.F_k_gain-self.F_k_loss)
                            *self.fft5(self.ifft5(self.exp*f_hat[...,None,None])*f[...,None,None]),axis=(-1, -2))
        return np.real(self.ifft3(Q))
    
    def col_sep(self, f):
        # fft of f
        f_hat = self.fft3(f)
        # gain term
        Q_gain = self.s_w*np.sum(self.r_w*self.F_k_gain
                                 *self.fft5(self.ifft5(self.exp*f_hat[...,None,None])*f[...,None,None]),axis=(-1, -2))
        # loss term
        Q_loss = 4*pi*np.sum(self.r_w*self.F_k_loss
                        *self.fft4(self.ifft4(self.sinc*f_hat[...,None])*f[...,None]),axis=(-1))
        return np.real(self.ifft3(Q_gain - Q_loss))
    
    def laplacian(self, f):
        return np.real(self.ifft3(self.lapl*self.fft3(f)))
    
    def col_heat_full(self, f_hat, eps):
        # ifft
        f = self.ifft3(f_hat)
        # convolution and quadrature
        Q = self.s_w*np.sum(self.r_w*(self.F_k_gain-self.F_k_loss)
                            *self.fft5(self.ifft5(self.exp*f_hat[...,None,None])*f[...,None,None]),axis=(-1, -2))
        return Q/(4*pi) + eps*self.lapl*f_hat
    
    def col_heat_sep(self, f_hat, eps):
        # ifft of f
        f = self.ifft3(f_hat)
        # gain term
        Q_gain = self.s_w*np.sum(self.r_w*self.F_k_gain
                                 *self.fft5(self.ifft5(self.exp*f_hat[...,None,None])*f[...,None,None]),axis=(-1, -2))
        # loss term
        Q_loss = 4*pi*np.sum(self.r_w*self.F_k_loss
                        *self.fft4(self.ifft4(self.sinc*f_hat[...,None])*f[...,None]),axis=(-1))
        return (Q_gain - Q_loss)/(4*pi) + eps*self.lapl*f_hat