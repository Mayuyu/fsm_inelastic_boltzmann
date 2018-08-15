import numpy as np
from utility import RK3
from fast_spec_col_2d import FastSpectralCollison2D


class TestModule2D:
    def __init__(self, config, initial, N, e, eps=0):
        S = config.domain_config.S
        self.L = eval(config.domain_config.L)
        self.N = config.domain_config.N
        self.e, self.eps = e, eps
        self.f0 = initial
        self.Q = []
        for n in N:
            self.Q.append(FastSpectralCollison2D(config, e=e, N=n))
            
    def ext_T(self, N_index, tfinal):
        Q = self.Q[N_index]
        v, v_norm, dv = Q.v, Q.v_norm, Q.dv
        T_0 = 0.5*np.sum(self.f0(v)*v_norm)*dv**2
#         T_0 = 1.425
        return (T_0 - 8*self.eps/(1-self.e**2))*np.exp(-(1-self.e**2)*tfinal/4) + 8*self.eps/(1-self.e**2)
    
    def solve(self, RK, dt, tfinal, N_index, method):
        Q = self.Q[N_index]
        v, v_norm, dv = Q.v, Q.v_norm, Q.dv
        f_hat = Q.fft2(self.f0(v))
        
        for _ in np.arange(0, tfinal, dt):
            f_hat = RK(f_hat, getattr(Q, 'col_heat_hat_'+ method), self.eps, dt)    
        f = np.real(Q.ifft2(f_hat))
        return f, 0.5*np.sum(f*v_norm)*dv**2
            
    def dt_test(self, Dt, method='sep', RK=RK3, N_index=3, tfinal=2):
        num_T, num_f = [], []
        for dt in Dt:
            f, T = self.solve(RK, dt, tfinal, N_index, method)
            num_T.append(T)
            num_f.append(f)
        return np.abs(np.asarray(num_T) - self.ext_T(N_index, tfinal)), num_f, num_T
    
    def N_test(self, dt, method='sep', RK=RK3, tfinal=2):   
        num_T, num_f, ext_T = [], [], []
        for N_index in [0,1,2,3,4]:
            f, T = self.solve(RK, dt, tfinal, N_index, method)
            num_T.append(T)
            num_f.append(f)
            ext_T.append(self.ext_T(N_index, tfinal))
        return np.abs(np.asarray(num_T) - np.asarray(ext_T)), num_f, num_T