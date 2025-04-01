#!/usr/bin/env python3

import numpy as np

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

ship_config = ShipConfig()