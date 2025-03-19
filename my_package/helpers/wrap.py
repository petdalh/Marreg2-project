#!/usr/bin/env python3

import numpy as np

def wrap(yaw):

    return ((yaw + np.pi) % (2 * np.pi)) - np.pi