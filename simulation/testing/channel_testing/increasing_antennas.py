
import matplotlib.pyplot as plt
import seaborn
from src.network_simulation_tools import *

#   Initialize System Parameters
size = 100
rows = size
cols = size
sigma = 1   # Relative attenuation of IRS portion
rho = 1     # S/R
phi = 1    # phi=1 is full rank line of sight.  L/T
#   Plot Theoretical Distribution and find corresponding capacity
numeric_capacity_list = []
water_fill_capacity = []
G_fill_capacity = []
H_fill_capacity = []
param_list = [5, 10, 50, 100]


for val in param_list:
    #   Now check histogram of simulations
    rows = cols = val
    irs = []
    irs_svd = []
    water_capacity = []
    H_capacity = []
    G_capacity = []
    capacities = []
    cross_sum = 0
    main_sum = 0
    los_rank_matrix = reduced_rank(rows, phi)
    average = 1
    for i in range(average):
        H = sigma*c_rand(rows, cols) @ c_rand(rows, cols)
        G = c_rand(rows, cols)@los_rank_matrix
        GG = G @ np.conj(G.T)
        HH = H @ np.conj(H.T)
        GH = G @ np.conj(H.T)
        HG = H @ np.conj(G.T)
        # total_cov = GH + HG + GG + HH
        # total_channel = H + G
        total_cov = GG
        total_channel = G
        capacities.append(capacity(total_cov, 1/cols))
        water_capacity.append(capacity_water_filled(water_filling(total_channel, 1)))
        H_capacity.append(capacity_water_filled(water_filling(H, 1)))
        G_capacity.append(capacity_water_filled(water_filling(G, 1)))

# Compare Estimated to True Capacity
    numeric_capacity_list.append(np.average(capacities))
    water_fill_capacity.append(np.average(water_capacity))
    G_fill_capacity.append(np.average(G_capacity))
    H_fill_capacity.append(np.average(H_capacity))

#   Plot results
fig, ax = plt.subplots()
plt.title("Capacity Vs. Num Antennas")
# plt.title("Capacity Vs. IRS Path Attenuation")
ax.plot(param_list, numeric_capacity_list, label='equal power', c='r')
ax.plot(param_list, water_fill_capacity, label='water filled capacity', c='g')
# ax.plot(param_list, H_fill_capacity, label='IRS channel')
# ax.plot(param_list, G_fill_capacity, label='LOS channel')

ax.set_ylabel('Capacity (Bits)')
# ax.set_xlabel('Relative Attenuation Coefficient')
ax.set_xlabel('Number Antennas')
plt.legend(loc="upper left")
ax.grid(True, which='both')
seaborn.despine(ax=ax, offset=0)
plt.show()


