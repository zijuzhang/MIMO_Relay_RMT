# A test for looking at impact of adding correlation through concatenated channels on the channel capacity
# Takeaway:
# If power is allocated equally the ave capacity doesn't seem to be affected however variance increase
# with increasing correlation. However if water-filling
# is used then the correlated channels seem to have increasing capacity as correlation increases

import matplotlib.pyplot as plt
from src.fixed_point import *
from src.lin_alg import *
from src.network_simulation_tools import water_filling
import seaborn
from scipy.linalg import toeplitz


size = 100
bins = int(size/2)
irs_cor = []
rank = 100
phi = rank/size
correlation_block_sizes = [1, 5, 25, 100]
correlated_capacity_averages = np.zeros(len(correlation_block_sizes))

fig = plt.figure()
ave_plt = fig.add_subplot(2, 1, 1)
var_plt = fig.add_subplot(2, 1, 2)
average = 100
sizes = [10, 20, 50, 100]
correlations = [.5, .8, .9, .98]
for cor_rho in correlations:
    capacity_variance = []
    capacity_average = []
    for size in sizes:
        t_correlation_matrix = exponential_correlation(size, cor_rho)
        r_correlation_matrix = exponential_correlation(size, cor_rho)
        cor_cap = []
        for i in range(average):
            G = c_rand(size, size)
            H_cor = r_correlation_matrix@G@t_correlation_matrix
            HH_cor = H_cor@hermetian(H_cor)
            # e_val_cor = np.linalg.eigvalsh(HH_cor)
            # irs_cor.append(np.real(e_val_cor))
            # cor_trace.append(np.trace(HH_cor))
            cor_cap.append(capacity(HH_cor, 1/size))
            # cor_cap.append(capacity_water_filled(water_filling(H_cor, 1)))
        capacity_average.append(np.average(cor_cap))
        capacity_variance.append(np.var(cor_cap))

    var_plt.plot(sizes, capacity_variance, '-o', label=f'Capacity Variance: {cor_rho}')
    ave_plt.plot(sizes, capacity_average, '-o', label=f'Capacity Average: {cor_rho}')


# ave_plt.title("Capacity Average")
# plt.title("Capacity Variance")
ave_plt.grid(True, which='both')
seaborn.despine(ax=ave_plt, offset=0)
ave_plt.legend(loc="upper right")

var_plt.grid(True, which='both')
seaborn.despine(ax=var_plt, offset=0)
var_plt.legend(loc="upper right")
plt.show()

