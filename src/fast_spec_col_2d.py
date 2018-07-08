# fast_spec_col_2d

import numpy as np
import pyfftw
from scipy import special
from math import pi

class FastSpectralCollison2D:
    
    def __init__(self, e, gamma, L, N, R, N_R, M=8):
        # legendre quadrature
        x, w = np.polynomial.legendre.leggauss(N_R)
        r = 0.5*(x + 1)*R
        self.r_w = 0.5*R*w
        # circular points and weight
        self.s_w = 2*pi/M
        m = np.arange(0, 2*pi, self.s_w)
        # index
        k = np.fft.fftshift(np.arange(-N/2,N/2))
        # dot with index
        rkg = (k[:,None,None]*np.cos(m) + k[:,None]*np.sin(m))[...,None]*r
        # norm of index
        k_norm = np.sqrt(k[:,None]**2 + k**2)
        # gain kernel
        self.F_k_gain = 2*pi*r**(gamma+1)*(np.exp(0.25*1j*pi*(1+e)*rkg/L)
                                           *special.jn(0, 0.25*pi*(1+e)*r*k_norm[...,None,None]/L))
        # loss kernel
        self.F_k_loss = 2*pi*r**(gamma+1)
        # exp for fft
        self.exp = np.exp(-1j*pi*rkg/L)
        # j0
        self.j0 = special.jn(0, pi*r*k_norm[...,None]/L)
        # laplacian
        self.lapl = -pi**2/L**2*k_norm**2
        
        # pyfftw planning of (N, N)
        array_2d = pyfftw.empty_aligned((N, N), dtype='complex128')
        self.fft2 = pyfftw.builders.fft2(array_2d, overwrite_input=True,
                                   planner_effort='FFTW_ESTIMATE', threads=20, avoid_copy=True)
        self.ifft2 = pyfftw.builders.ifft2(array_2d, overwrite_input=True,
                                     planner_effort='FFTW_ESTIMATE', threads=20, avoid_copy=True)
        # pyfftw planning of (N, N, N_R)
        array_3d = pyfftw.empty_aligned((N, N, N_R), dtype='complex128')
        self.fft3 = pyfftw.builders.fftn(array_3d, axes=(0,1), overwrite_input=True,
                                   planner_effort='FFTW_ESTIMATE', threads=20, avoid_copy=True)
        self.ifft3 = pyfftw.builders.ifftn(array_3d, axes=(0,1), overwrite_input=True,
                                     planner_effort='FFTW_ESTIMATE', threads=20, avoid_copy=True)
        # pyfftw planning of (N, N, M, N_R)
        array_4d = pyfftw.empty_aligned((N, N, M, N_R), dtype='complex128')
        self.fft4 = pyfftw.builders.fftn(array_4d, axes=(0,1), overwrite_input=True,
                                   planner_effort='FFTW_ESTIMATE', threads=20, avoid_copy=True)
        self.ifft4 = pyfftw.builders.ifftn(array_4d, axes=(0,1), overwrite_input=True,
                                     planner_effort='FFTW_ESTIMATE', threads=20, avoid_copy=True)
        
    def col_full(self, f):
        # fft of f
        f_hat = self.fft2(f)
        # convolution and quadrature
        Q = self.s_w*np.sum(self.r_w*(self.F_k_gain-self.F_k_loss)
                            *self.fft4(self.ifft4(self.exp*f_hat[...,None,None])*f[...,None,None]), axis=(-1, -2))
        return np.real(self.ifft2(Q))
    
    def col_sep(self, f):
        # fft of f
        f_hat = self.fft2(f)
        # gain term
        Q_gain = self.s_w*np.sum(self.r_w*self.F_k_gain
                                 *self.fft4(self.ifft4(self.exp*f_hat[...,None,None])*f[...,None,None]), axis=(-1, -2))
        # loss term
        Q_loss = 2*pi*np.sum(self.r_w*self.F_k_loss
                        *self.ifft3(self.j0*f_hat[...,None])*f[...,None], axis=(-1))
        return np.real(self.ifft2(Q_gain) - Q_loss)
    
    def laplacian(self, f):
        return np.real(self.ifft2(self.lapl*self.fft2(f)))
    
    def col_heat_hat_full(self, f_hat, eps):
        # ifft
        f = self.ifft2(f_hat)
        # convolution and quadrature
        Q = self.s_w*np.sum(self.r_w*(self.F_k_gain-self.F_k_loss)
                            *self.fft4(self.ifft4(self.exp*f_hat[...,None,None])*f[...,None,None]), axis=(-1, -2))
        return Q/(2*pi) + eps*self.lapl*f_hat
    
    def col_heat_hat_sep(self, f_hat, eps):
        # ifft of f
        f = self.ifft2(f_hat)
        # gain term
        Q_gain = self.s_w*np.sum(self.r_w*self.F_k_gain
                                 *self.fft4(self.ifft4(self.exp*f_hat[...,None,None])*f[...,None,None]), axis=(-1, -2))
        # loss term
        Q_loss = 2*pi*np.sum(self.r_w*self.F_k_loss
                        *self.fft3(self.ifft3(self.j0*f_hat[...,None])*f[...,None]), axis=(-1))
        return (Q_gain - Q_loss)/(2*pi) + eps*self.lapl*f_hat
