from src.network_simulation_tools import *
from src.network_simulation_tools import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn

fig, ax = plt.subplots()
plt.title("")
surface_size = 500
bins = int(surface_size/2)
original_network = [1]
list_irs = [1, 1]
capacity_water = []
capacity_water_phase = []
for num_irs in [1]:
    list_irs = original_network + [num_irs]
    # list_irs = list_irs + [1]
    net = Network(surface_size, copy.copy(list_irs))
    optimized_powers_1 = water_filling(net.get_channel_covariance(), 1)
    cov_water = net.get_transmit_covariance(np.diag(optimized_powers_1))
    e_val_water, e_vec_water = np.linalg.eig(cov_water)
    AED_water, AED_bins_water = np.histogram(np.asarray(e_val_water), bins=bins)
    capacity_water.append(net.get_capacity())

    #   Now optimize phases and repeat above steps with all else constant
    net.update_phases()
    optimized_powers_2 = water_filling(net.get_channel_covariance(), 1)
    cov_water_phase = net.get_transmit_covariance(np.diag(optimized_powers_2))
    e_val_water_phase, e_vec_water_phase = np.linalg.eig(cov_water_phase)
    AED_water_phase, AED_bins_water_phase = np.histogram(np.asarray(e_val_water_phase), bins=bins)
    capacity_water_phase.append(net.get_capacity())

    ax.scatter(AED_bins_water[1::], AED_water/surface_size, marker='>', label=f'Non-optimized Phase: {num_irs}')
    ax.scatter(AED_bins_water_phase[1::], AED_water_phase/surface_size, marker="*", label=f'Optimized Phase: {num_irs}')


print(f"water filling phase 1{capacity} \n")
print(f"water filling phase 2 {capacity_water} \n")





plt.legend(loc="upper right")
ax.grid(True, which='both')
seaborn.despine(ax=ax, offset=0)

plt.show()
