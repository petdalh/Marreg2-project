#!/usr/bin/env python3
import numpy as np
from template_observer.wrap import wrap

def create_R(psi):
    """Create rotation matrix from heading angle."""
    return np.array([
        [np.cos(psi), -np.sin(psi), 0],
        [np.sin(psi), np.cos(psi), 0],
        [0, 0, 1]
    ])

def luenberger(
        x_hat_prev: np.ndarray, # Previous estimated state vector
        eta: np.ndarray, # Measured position
        tau: np.ndarray, #
        L1: np.ndarray,
        L2: np.ndarray,
        L3: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    # System parameters
    M = np.array([[16.0, 0.0, 0.0], [0.0, 24.0, 0.53], [0.0, 0.53, 2.8]])
    D = np.array([[0.66, 0.0, 0.0], [0.0, 1.3, 2.8], [0.0, 0.0, 1.9]])
    T_b = np.diag([124, 124, 124])
    dt = 0.1
    
    # Extract previous state estimates
    eta_hat = x_hat_prev[0:3].reshape(3, 1)
    nu_hat = x_hat_prev[3:6].reshape(3, 1)
    bias_hat = x_hat_prev[6:9].reshape(3, 1)
    
    # Create rotation matrix
    psi_hat = wrap(eta_hat[2, 0])
    R = create_R(psi_hat)
    
    # y_tilde = y - ŷ = y - η̂
    y_tilde = eta - eta_hat
        
    # A matrix - system dynamics
    M_inv = np.linalg.inv(M)
    T_b_inv = np.linalg.inv(T_b)
    
    A = np.block([
        [np.zeros((3, 3)), R, np.zeros((3, 3))],
        [np.zeros((3, 3)), -M_inv @ D, M_inv],
        [np.zeros((3, 3)), np.zeros((3, 3)), -T_b_inv]
    ])
    
    # B matrix - control input mapping
    B = np.vstack([
        np.zeros((3, 3)),
        M_inv,
        np.zeros((3, 3))
    ])
    
    # C matrix - measurement feedback
    C = np.vstack([
        L1,
        M_inv @ L2 @ R.T,
        L3 @ R.T
    ])
    
    # Stack the state vector
    x = np.vstack([eta_hat, nu_hat, bias_hat])
    
    # Calculate state derivatives: ẋ = Ax + Bτ + Cỹ
    x_dot = A @ x + B @ tau + C @ y_tilde
    
    # Euler integration to update state estimates
    x_new = x + x_dot * dt
    
    # Extract updated state estimates
    eta_hat_new = x_new[0:3]
    nu_hat_new = x_new[3:6]
    bias_hat_new = x_new[6:9]
    
    return eta_hat_new, nu_hat_new, bias_hat_new