#!/usr/bin/env python3

import numpy as np

def PD_FF_controller(observer, reference, P_gain: float, D_gain: float) -> np.ndarray:
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
    
    K_p = np.diag([P_gain, P_gain, P_gain])  
    K_d = np.diag([D_gain, D_gain, D_gain])
    
    tau_pd = -K_p @ e - K_d @ nu_hat
    
    tau = tau_pd - bias_hat
    
    return tau

class PID_controller:
    def __init__(self):
        self.e_integral = np.zeros((3, 1))
        self.dt = 0.1

    def update(self, observer, reference, P_gain, I_gain, D_gain):

        # Extract observer and reference values
        eta_hat = np.array(observer.eta).reshape(3, 1)
        nu_hat = np.array(observer.nu).reshape(3, 1)
        eta_ref = np.array(reference.eta_d).reshape(3, 1)

        # Compute orientation error
        psi = eta_hat[2, 0]
        R_transpose = np.array([
            [np.cos(psi), np.sin(psi), 0],
            [-np.sin(psi), np.cos(psi), 0],
            [0, 0, 1]
        ])
        e = R_transpose @ (eta_hat - eta_ref)

        # Update integral term (only if `dt` is valid)
        self.e_integral += e * self.dt

        # PID gains
        K_p = np.diag([P_gain, P_gain, P_gain])
        K_i = np.diag([I_gain, I_gain, I_gain])
        K_d = np.diag([D_gain, D_gain, D_gain])

        K_p[3][3] *= 100

        # Compute control output
        tau_pid = -K_p @ e - K_i @ self.e_integral - K_d @ nu_hat
        return tau_pid