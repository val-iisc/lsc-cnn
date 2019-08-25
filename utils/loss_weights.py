"""
main.py: Code to calculate box weights. 
Authors       : svp
"""

import numpy as np
np.set_printoptions(suppress=True)

def compute_box_weights(hist):
    
    hist = np.float32(hist)
    for i in range(hist.shape[0]):
        hist[i] /= (4 ** (hist.shape[0] - i - 1))
    hist_copy = hist.copy()
    num_channels = hist.shape[1]
    
    c_sum = np.sum(hist[:, 0:num_channels-1], axis=1)
    c_min = np.min(c_sum) 
    z_weights = c_min / c_sum
    z_weights = np.expand_dims(z_weights, axis=1)
    z_weights = np.repeat(z_weights, num_channels, axis=1)

    c_0 = hist[:, 3]
    c_0 = np.expand_dims(c_0, axis=1)
    c_0 = np.repeat(c_0, num_channels, axis=1)
    c_b =hist
    weight_up = (c_0 / c_b)
    tens = np.ones((weight_up.shape)) * 10

    minimum = np.minimum(tens, weight_up)
    alpha = minimum * z_weights
    
    return alpha
