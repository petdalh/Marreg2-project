import numpy as np

# Define global parameters (for example):
p0 = np.array([5.0, 0.0])
p1 = np.array([0.0, 0])
U_ref = 1.0
psi_ref = None

# Precompute path vectors:
pd_s = p1 - p0
path_length = np.linalg.norm(pd_s)
# Constant heading if psi_ref not given:
psi_d_const = psi_ref if psi_ref is not None else np.arctan2(pd_s[1], pd_s[0])

def straight_line():
    """
    Returns:
       eta_d   = [x_d, y_d, psi_d]
       eta_ds  = [x'_d, y'_d, psi'_d]
       eta_ds2 = [x''_d, y''_d, psi''_d]
    """
    # Example path parameter in [0,1]
    s = 0.5

    # Position on line
    pd = (1 - s)*p0 + s*p1

    # Heading
    psi_d = psi_d_const

    # Build outputs
    eta_d = np.array([pd[0], pd[1], psi_d])

    # First derivative w.r.t. path parameter s
    pd_s_   = pd_s
    psi_d_s = 0.0
    eta_ds  = np.array([pd_s_[0], pd_s_[1], psi_d_s])

    # Second derivative is zero on a straight line
    pd_s2_   = np.zeros_like(pd_s_)
    psi_d_s2 = 0.0
    eta_ds2  = np.array([pd_s2_[0], pd_s2_[1], psi_d_s2])

    return eta_d, eta_ds, eta_ds2

def update_law():
    """
    Returns:
       w, v_s, v_ss
    """
    # Example constant rate of change for s
    w = 0.1
    v_s = 0.0
    v_ss = 0.0
    return w, v_s, v_ss
