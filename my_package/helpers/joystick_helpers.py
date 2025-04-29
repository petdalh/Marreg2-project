#!/usr/bin/env python3
import numpy as np

def handle_controller_input(axes):
    max_surge = 2.0
    max_sway = 2.0
    max_yaw = 100.0

    # Map trigger values from [1, -1] to [0, 1]
    left_trigger = (1 - axes[5]) / 2
    right_trigger = (1 - axes[2]) / 2
    
    if left_trigger > 0:  
        tau_cmd = np.array([
            max_surge * axes[1],
            max_sway * axes[0],
            -max_yaw * left_trigger  
        ])
    elif right_trigger > 0:  
        tau_cmd = np.array([
            max_surge * axes[1],
            max_sway * axes[0],
            max_yaw * right_trigger  
        ])
    else:  
        tau_cmd = np.array([
            max_surge * axes[1],
            max_sway * axes[0],
            0
        ])
    return tau_cmd