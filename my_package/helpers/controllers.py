#!/usr/bin/env python3

import numpy as np

def PID_controller(observer, reference, P_gain: float, I_gain: float, D_gain: float) -> np.ndarray:
    eta_hat = np.array(observer.eta).reshape(3, 1)  
    nu_hat = np.array(observer.nu).reshape(3, 1)    
    bias_hat = np.array(observer.bias).reshape(3, 1)
    
    eta_ref = np.array(reference.eta_d).reshape(3, 1)
    
    psi = eta_hat[2, 0]
    
    R_transpose = np.array([
        [np.cos(psi), np.sin(psi), 0],
        [-np.sin(psi), np.cos(psi), 0],
        [0, 0, 1]
    ])
    
    e = R_transpose @ (eta_hat - eta_ref)
    
    K_p = np.diag([P_gain, P_gain, P_gain*0.5])  
    K_d = np.diag([D_gain, D_gain, D_gain*0.5])
    
    tau_pd = -K_p @ e - K_d @ nu_hat
    
    tau = tau_pd - bias_hat
    
    return tau