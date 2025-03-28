#!/usr/bin/env python3
import numpy as np
from .wrap import wrap
from .create_R import create_R

class ShipConfig:
    """Configuration object holding system parameters and utility functions"""
    
    def __init__(self):
        # System parameters
        self.M = np.array([[16.0, 0.0, 0.0], [0.0, 24.0, 0.53], [0.0, 0.53, 2.8]])
        self.D = np.array([[0.66, 0.0, 0.0], [0.0, 1.3, 2.8], [0.0, 0.0, 1.9]])
        self.T_b = np.diag([124, 124, 124])
        self.dt = 0.1
        
        # Precomputed values
        self.M_inv = np.linalg.inv(self.M)
        self.T_b_inv = np.linalg.inv(self.T_b)


# Create a singleton instance of the ship configuration
ship_config = ShipConfig()


def extract_states(x_hat_prev):
    """Extract state components from the state vector"""
    eta_hat = x_hat_prev[0:3].reshape(3, 1)
    nu_hat = x_hat_prev[3:6].reshape(3, 1)
    bias_hat = x_hat_prev[6:9].reshape(3, 1)
    return eta_hat, nu_hat, bias_hat


def create_rotation_matrix(eta_hat):
    """Create rotation matrix from heading"""
    psi_hat = wrap(eta_hat[2, 0])
    return create_R(psi_hat)


def luenberger(x_hat_prev, eta, tau, L1, L2, L3, config=ship_config):
    """
    Update state estimate using Luenberger observer with configuration object
    """
    # Extract previous state estimates
    eta_hat, nu_hat, bias_hat = extract_states(x_hat_prev)
    
    # Create rotation matrix
    R = create_rotation_matrix(eta_hat)
    
    # y_tilde = y - ŷ = y - η̂
    y_tilde = eta - eta_hat
        
    # A matrix - system dynamics
    A = np.block([
        [np.zeros((3, 3)), R, np.zeros((3, 3))],
        [np.zeros((3, 3)), -config.M_inv @ config.D, config.M_inv],
        [np.zeros((3, 3)), np.zeros((3, 3)), -config.T_b_inv]
    ])
    
    # B matrix - control input mapping
    B = np.vstack([
        np.zeros((3, 3)),
        config.M_inv,
        np.zeros((3, 3))
    ])
    
    # C matrix - measurement feedback
    C = np.vstack([
        L1,
        config.M_inv @ L2 @ R.T,
        L3 @ R.T
    ])
    
    # Stack the state vector
    x = np.vstack([eta_hat, nu_hat, bias_hat])
    
    # Calculate state derivatives: ẋ = Ax + Bτ + Cỹ
    x_dot = A @ x + B @ tau + C @ y_tilde
    
    # Euler integration to update state estimates
    x_new = x + x_dot * config.dt
    
    # Extract updated state estimates
    eta_hat_new = x_new[0:3]
    nu_hat_new = x_new[3:6]
    bias_hat_new = x_new[6:9]
    
    return eta_hat_new, nu_hat_new, bias_hat_new


def dead_reckoning(x_hat_prev, tau, config=ship_config):
    """
    Update state estimate using dead reckoning with configuration object
    """
    # Extract previous state estimates
    eta_hat, nu_hat, bias_hat = extract_states(x_hat_prev)
    
    # Create rotation matrix
    R = create_rotation_matrix(eta_hat)
    
    # Calculate derivatives based on ODM
    eta_dot = R @ nu_hat
    nu_dot = config.M_inv @ (-config.D @ nu_hat + R.T @ bias_hat + tau)
    # b_dot = np.zeros((3, 1)) # Bias is constant

    # If bias is not constant
    alpha = 0.1  # Small drift coefficient
    b_dot = alpha * bias_hat
    
    # Euler integration to update state estimates
    eta_hat_new = eta_hat + eta_dot * config.dt
    nu_hat_new = nu_hat + nu_dot * config.dt
    bias_hat_new = bias_hat + b_dot * config.dt
    
    return eta_hat_new, nu_hat_new, bias_hat_new