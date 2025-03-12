
import numpy as np

def handle_controller_input(axes):
    max_surge = 2.0
    max_sway = 2.0
    max_yaw = 1.0

    if axes[2] < 1:
        tau_cmd = np.array([
            max_surge * axes[1],
            max_sway * axes[0],
            -max_yaw * axes[2] * 100
        ])
    elif axes[5] < 1:
        tau_cmd = np.array([
            max_surge * axes[1],
            max_sway * axes[0],
            max_yaw * axes[5] * 100
        ])
    else:
        tau_cmd = np.array([
            max_surge * axes[1],
            max_sway * axes[0],
            0
        ])
    return tau_cmd
