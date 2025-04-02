#!/usr/bin/env python3
import numpy as np


def thruster_allocation_extended(tau: np.ndarray) -> np.ndarray:
    # Thruster positions (x_i, y_i).
    x1, y1 = (-41.5, -7)  
    x2, y2 = (-41.5, 7)   
    x3, y3 = (37, 0)

    # Thruster gains (F_i = k_i * u_i)
    K = np.array([1.0, 1.0, 1.0], dtype=float)

    # Weight matrix W (here just the identity)
    W = np.eye(5)

    # Feed-forward f_d (size 5).  Often zero by default
    f_d = np.zeros(5)
    
    B = np.array([
        [0 ,1, 0, 1, 0],  # surge
        [1, 0, 1, 0, 1],  # sway
        [x3, -y1, x1, -y2, x2]  # yaw
    ], dtype=float)

    
    # Compute the weighted pseudoinverse B_W^†
    W = np.eye(3)  # For now, just identity
    W_sqrt = np.sqrt(W)  # Or np.diag(np.sqrt(np.diag(W))) if needed
    B_w = W_sqrt @ B
    B_w_pinv = np.linalg.pinv(B_w)  


    # f* = B_W^† tau_cmd + (I - B_W^† B) f_d = B_W^† tau_cmd
    Q_w = np.eye(5) - B_w_pinv @ B
    f_star = B_w_pinv @ tau + Q_w @ f_d

    # Parse out X1*, Y1*, X2*, Y2*, Y3*
    Y3_star, X1_star, Y1_star, X2_star, Y2_star = f_star

    
    # Convert to thruster magnitudes/angles for thrusters 1 & 2
    F1_star = np.hypot(X1_star, Y1_star)
    F2_star = np.hypot(X2_star, Y2_star)

    alpha1_star = np.arctan2(Y1_star, X1_star)
    alpha2_star = np.arctan2(Y2_star, X2_star)

    # Third thruster
    F3_star = Y3_star

    # Convert to thruster inputs u_i = F_i / k_i
    u1_star = F1_star / K[0]
    u2_star = F2_star / K[1]
    u3_star = F3_star / K[2]

    # Clamp each thruster input to [0, 1]:
    u3_cmd = max(-1.0, min(u3_star, 1.0))  # Allow negative for lateral thruster
    u1_cmd = max(-1.0, min(u1_star, 1.0))   # Keep positive for VSP thrusters
    u2_cmd = max(-1.0, min(u2_star, 1.0))

    # Wrap angles to [-pi, pi], if needed
    def wrap_angle(a):
        return (a + np.pi) % (2.0 * np.pi) - np.pi

    alpha1_cmd = wrap_angle(alpha1_star)
    alpha2_cmd = wrap_angle(alpha2_star)

    # 5x1 vector: [u1, u2, u3, alpha1, alpha2]
    return np.array([u3_cmd, u1_cmd, u2_cmd, alpha1_cmd, alpha2_cmd])