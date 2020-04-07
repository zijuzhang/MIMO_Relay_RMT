from src.network_simulation_tools import *
from src.network_simulation_tools import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn

test = water_filling(np.eye(10), 10)
bins = 300
surface_size = 500
net = Network(surface_size, [1, 1, 1])
cov = net.get_covariance()
e_val, e_vec = np.linalg.eig(cov)
AED, AED_bins = np.histogram(np.asarray(e_val), bins=bins)


fig, ax = plt.subplots()
plt.title("Eigen Vector Comparison: i.i.d, $\mathbb{N}(0,1/N)$, H = H_1 \Phi H_2")
ax.plot(AED_bins[1::], AED/surface_size, label='cross no IRS')
plt.legend(loc="upper right")
ax.grid(True, which='both')
seaborn.despine(ax=ax, offset=0)

plt.show()
