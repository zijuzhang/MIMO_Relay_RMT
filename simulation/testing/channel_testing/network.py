from src.network_simulation_tools import *
from src.network_simulation_tools import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn

fig, ax = plt.subplots()
plt.title("Number of IRS (N=1): i.i.d, $\mathbb{N}(0,1/N)$, H = H_1 \Phi H_2")
surface_size = 500
bins = int(surface_size/2)
original_network = [1]
list_irs = [1]
capacity = []
secrecy_capacities = []
capacity_water = []
secrecy_capacities_water = []
for num_irs in [8, 16, 32]:
    list_irs = original_network + [num_irs]
    # list_irs = list_irs + [1]
    net = Network(surface_size, copy.copy(list_irs), eavesdropper=True)
    transmit_covariance = water_filling(net.channel, 1)
    e_val_water, e_vec_water = np.linalg.eig(net.covariance)
    AED_water, AED_bins_water = np.histogram(np.asarray(e_val_water), bins=bins)
    secrecy_capacities_water.append(net.get_capacity(transmit_covariance, secrecy=True))
    capacity_water.append(net.get_capacity(transmit_covariance))

    # e_val, e_vec = np.linalg.eig(cov)
    # AED, AED_bins = np.histogram(np.asarray(e_val), bins=bins)
    secrecy_capacities.append(net.get_capacity(np.eye(surface_size)/surface_size, secrecy=True))
    capacity.append(net.get_capacity(np.eye(surface_size)/surface_size))

    # ax.plot(AED_bins[1::], AED/surface_size, label=f'Number of IRS: {num_irs}')
    ax.plot(AED_bins_water[1::], AED_water/surface_size, label=f'Water Filling: {num_irs}')


print(f"no water filling {capacity} \n")
print(f"water filling {capacity_water} \n")
print(f"sec no water filling {secrecy_capacities} \n")
print(f"sec water filling {secrecy_capacities_water} \n")




plt.legend(loc="upper right")
ax.grid(True, which='both')
seaborn.despine(ax=ax, offset=0)

plt.show()
