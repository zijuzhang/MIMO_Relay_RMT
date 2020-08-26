import numpy as np
from src.lin_alg import *



def ML_MIMO(received_pilots, pilots):
    #   Assume that the pilot matrix is orthogonal with respect to the vectors at each Tx antenna.
    #   Basically we don't want any of our equations to be linearly dependent.
    return hermetian(np.linalg.pinv(pilots@hermetian(pilots))@pilots@hermetian(received_pilots))
