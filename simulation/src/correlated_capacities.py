import numpy as np


def loyka_upper_bound(num_antenna, correlation_rate, snr):
    return np.log2(1+(snr/num_antenna)*(1-pow(correlation_rate, 2)))

