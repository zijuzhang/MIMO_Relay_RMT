from src.network_simulation_tools import *
from src.network_simulation_tools import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn

fig, ax = plt.subplots()
plt.title("Number of IRS (N=1): i.i.d, $\mathbb{N}(0,1/N)$, H = H_1 \Phi H_2")
surface_size = 300
bins = int(surface_size/2)
list_irs = [1, 1]
secrecy_capacities = []
secrecy_capacities_water = []
for num_irs in [2, 4]:
    list_irs += [1]
    net = Network(surface_size, list_irs, eavesdropper=True)
    optiized_powers = water_filling(net.get_channel_covariance(), 10)
    cov_water = net.get_transmit_covariance(np.diag(optiized_powers))
    e_val_water, e_vec_water = np.linalg.eig(cov_water)
    AED_water, AED_bins_water = np.histogram(np.asarray(e_val_water), bins=bins)
    secrecy_capacities_water.append(net.get_capacity())

    cov = net.get_transmit_covariance(np.eye(surface_size)/surface_size)
    e_val, e_vec = np.linalg.eig(cov)
    AED, AED_bins = np.histogram(np.asarray(e_val), bins=bins)
    secrecy_capacities.append(net.get_capacity())


    ax.plot(AED_bins[1::], AED/surface_size, label=f'Number of IRS: {num_irs}')
    ax.plot(AED_bins_water[1::], AED_water/surface_size, label=f'Water Filling: {num_irs}')


print(f"no water filling {secrecy_capacities} \n")
print(f"water filling {secrecy_capacities_water} \n")


plt.legend(loc="upper right")
ax.grid(True, which='both')
seaborn.despine(ax=ax, offset=0)

plt.show()
