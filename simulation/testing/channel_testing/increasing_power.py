
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
SNRS = np.linspace(0, 40, 10)
param_list = np.power(10, SNRS/10)
for val in param_list:
    #   Now check histogram of simulations
    rows = cols = size
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
        # H = sigma*c_rand(rows, cols) @random_phase(rows)@ c_rand(rows, cols)
        H = sigma*c_rand(rows, cols) @ c_rand(rows, cols)
        G = c_rand(rows, cols)@los_rank_matrix
        # check = np.var(H)
        # check1 = np.var(G)
        F = c_rand(rows, cols)
        GG = G @ np.conj(G.T)
        HH = H @ np.conj(H.T)
        FF = F @ np.conj(F.T)
        GH = G @ np.conj(H.T)
        HG = H @ np.conj(G.T)
        GF = G @ np.conj(F.T)
        FG = F @ np.conj(G.T)
        total_irs = GH + HG + GG + HH
        # total_irs = GG
        total_channel = H + G
        Q_x = water_filling(total_channel, val)
        e_val_total_irs = np.linalg.eigvalsh(total_irs)
        s_val_total_irs = np.linalg.svd(total_channel)[1]
        irs.append(e_val_total_irs)
        irs_svd.append(s_val_total_irs)
        capacities.append(capacity(total_irs, val/cols))
        water_capacity.append(capacity_water_filled(Q_x))
        H_capacity.append(capacity_water_filled(water_filling(H, val)))
        G_capacity.append(capacity_water_filled(water_filling(G, val)))

# Compare Estimated to True Capacity
    numeric_capacity_list.append(np.average(capacities))
    water_fill_capacity.append(np.average(water_capacity))
    G_fill_capacity.append(np.average(G_capacity))
    H_fill_capacity.append(np.average(H_capacity))

#   Plot results
fig, ax = plt.subplots()
plt.title("Capacity Vs. Num Antennas")
# plt.title("Capacity Vs. IRS Path Attenuation")
ax.plot(SNRS, numeric_capacity_list, label='equal power', c='r')
ax.plot(SNRS, water_fill_capacity, label='water filled capacity', c='g')
ax.plot(SNRS, H_fill_capacity, label='IRS channel')
ax.plot(SNRS, G_fill_capacity, label='LOS channel')

ax.set_ylabel('Capacity (Bits)')
# ax.set_xlabel('Relative Attenuation Coefficient')
ax.set_xlabel('SNR (dB)')
plt.legend(loc="upper left")
ax.grid(True, which='both')
seaborn.despine(ax=ax, offset=0)
plt.show()


