from src.lin_alg import *
import matplotlib.pyplot as plt
from src.optimization import *

#   Setup System Parameters
num_tx = 16
num_rx = 10

#   Will want to look at how this optimization evolves as the number of paths increases (Need to normalize this then?)
num_reflectors = 128

#   Later will want to look at how increasing the number of pilots impacts the gains from phase optimization

#   Coherence Specific Parameters
opt_phase = optimize_phases(num_reflectors, None)
H1 = c_rand(num_reflectors, num_tx, var=1)
H2 = c_rand(num_rx, num_reflectors, var=1)
G = c_rand(num_rx, num_tx, var=1)
H_total = (G + H2@opt_phase@H1)
# One transmission
transmit_symbols = c_rand(num_rx, 1, var=1).flatten()
noise = c_rand(num_rx, 1, var=1).flatten()
#   For now use perfect CSI
csi_estimate = H_total
matched_filter = hermetian(csi_estimate)
transmit_symbols = matched_filter@transmit_symbols
received = H_total @ transmit_symbols + noise
#   Normalized MSE
MSE = 10*np.log(np.power(np.linalg.norm(received - transmit_symbols), 2))