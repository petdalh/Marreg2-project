#!/usr/bin/env python3
import numpy as np
from .wrap import wrap
from .create_R import create_R

def dead_reckoning(
        x_hat_prev: np.ndarray,
        tau: np.ndarray,
        dt: float = 0.1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    # System parameters (same as in Luenberger)
    M = np.array([[16.0, 0.0, 0.0], [0.0, 24.0, 0.53], [0.0, 0.53, 2.8]])
    D = np.array([[0.66, 0.0, 0.0], [0.0, 1.3, 2.8], [0.0, 0.0, 1.9]])
    T_b = np.diag([124, 124, 124])
    
    # Extract previous state estimates
    eta_hat = x_hat_prev[0:3].reshape(3, 1)
    nu_hat = x_hat_prev[3:6].reshape(3, 1)
    bias_hat = x_hat_prev[6:9].reshape(3, 1)
    
    # Create rotation matrix
    psi_hat = wrap(eta_hat[2, 0])
    R = create_R(psi_hat)
    
    # State derivatives using only model dynamics (no measurement correction)
    M_inv = np.linalg.inv(M)
    T_b_inv = np.linalg.inv(T_b)
    
    # Calculate derivatives based on ODM
    eta_dot = R @ nu_hat
    nu_dot = M_inv @ (-D @ nu_hat + R.T @ bias_hat + tau)
    b_dot = np.zeros((3, 1))  # Bias is constant
    
    # Euler integration to update state estimates
    eta_hat_new = eta_hat + eta_dot * dt
    nu_hat_new = nu_hat + nu_dot * dt
    bias_hat_new = bias_hat + b_dot * dt
    
    return eta_hat_new, nu_hat_new, bias_hat_new