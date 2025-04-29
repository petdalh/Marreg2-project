#!/usr/bin/env python3

import numpy as np
from .ship_config import ship_config

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
    
    K_p = np.diag([P_gain+1, P_gain+1, P_gain+1])  
    K_d = np.diag([D_gain+0.5, D_gain+0.5, D_gain]) 
    
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
        K_p = np.array([[6.31654682, 0.0, 0.0],
                        [0.0, 9.47482023, 0.20923561],
                        [0.0, 0.20923561, 1.10539569]])

        K_i = np.array([[0.39688034, 0.0, 0.0],
                        [0.0, 0.59532051, 0.01314666],
                        [0.0, 0.01314666, 0.06945406]])

        K_d = np.array([[19.44619298, 0.0, 0.0],
                        [0.0, 28.85928947, -2.13398236],
                        [0.0, 0.66601764, 1.61858377]])

        # Compute control output
        tau_pid = -K_p @ e - K_i @ self.e_integral - K_d @ nu_hat
        return tau_pid

def backstepping_controller(observer, reference, K1_gain, K2_gain, config=ship_config) -> np.ndarray:
    # Extract observer states
    eta_hat = np.array(observer.eta).reshape(3, 1)
    nu_hat = np.array(observer.nu).reshape(3, 1)
    bias_hat = np.array(observer.bias).reshape(3, 1)

    # Extract reference signals
    eta_d = np.array(reference.eta_d).reshape(3, 1)
    eta_ds = np.array(reference.eta_ds).reshape(3, 1)
    eta_ds2 = np.array(reference.eta_ds2).reshape(3, 1)
    w = reference.w
    v_s = reference.v_s
    v_ss = reference.v_ss

    K1 = np.diag([8, 8, 1])
    K2 = np.diag([20, 20, 15])

    # Rotation matrix (body to inertial)
    psi = eta_hat[2, 0]
    R_T = np.array([
        [np.cos(psi), np.sin(psi), 0],
        [-np.sin(psi), np.cos(psi), 0],
        [0, 0, 1]
    ])

    z1 = R_T @ (eta_hat - eta_d)

    # Backstepping math (see report)
    s_dot = w + v_s
    z1_dot = nu_hat - R_T @ eta_ds * s_dot - nu_hat[2, 0] * config.S @ z1
    alpha1 = -K1 @ z1 + R_T @ eta_ds * v_s
    psi_dot = nu_hat[2, 0]
    alpha1_dot = -K1 @ z1_dot - psi_dot * config.S @ R_T @ eta_ds * v_s
    z2 = nu_hat - alpha1

    # Final control law 
    tau = (
        config.M @ alpha1_dot
        + config.D @ nu_hat
        - K2 @ z2
        - bias_hat  
    )

    return tau.flatten()