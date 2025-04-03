#!/usr/bin/env python3

import numpy as np

def stationkeeping(eta_d) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eta_ds = np.array([0.0, 0.0, 0.0], dtype=float)
    eta_ds2 = np.array([0.0, 0.0, 0.0], dtype=float)

    return eta_d, eta_ds, eta_ds2