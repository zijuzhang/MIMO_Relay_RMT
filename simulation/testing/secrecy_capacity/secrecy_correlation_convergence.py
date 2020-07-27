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

irs_cor = []
rank = 100
fig = plt.figure()
ave_plt = fig.add_subplot(2, 1, 1)
var_plt = fig.add_subplot(2, 1, 2)
average = 10
sizes = [20, 50, 100, 200]
# average = 10
correlations = [0, .5, .9, .98]
for cor_rho in correlations:
    capacity_variance = []
    capacity_average = []
    for size in sizes:
        t_correlation_matrix = exponential_correlation(size, cor_rho)
        s_correlation_matrix = exponential_correlation(size, cor_rho)
        r_correlation_matrix = exponential_correlation(size, cor_rho)
        sec_cap = []
        for i in range(average):
            G = c_rand(size, size)
            G2 = c_rand(size, size)
            G3 = c_rand(size, size)
            # H_cor = r_correlation_matrix@G@t_correlation_matrix
            H_cor = r_correlation_matrix@G2@s_correlation_matrix@random_phase(size)@G@t_correlation_matrix
            H_cor_eve = r_correlation_matrix@G3@s_correlation_matrix@random_phase(size)@G@t_correlation_matrix
            HH_cor = H_cor@hermetian(H_cor)
            HH_cor_eve = H_cor_eve@hermetian(H_cor_eve)
            #   Optimize for the intended received
            powers = water_filling(H_cor, 1, vals=True)
            sec_cap.append(capacity_water_filled(H_cor@powers@hermetian(H_cor)) - capacity_water_filled(H_cor_eve@powers@hermetian(H_cor_eve)))
            # sec_cap.append(capacity(HH_cor, 1/size) - capacity(HH_cor_eve, 1/size))
        capacity_average.append(np.average(sec_cap))
        capacity_variance.append(np.var(sec_cap))

    var_plt.plot(sizes, capacity_variance, '-o', label=f'Correlation (rho): {cor_rho}')
    ave_plt.plot(sizes, capacity_average, '-o', label=f'Correlation (rho): {cor_rho}')


# ave_plt.title("Capacity Average")
# plt.title("Capacity Variance")
var_plt.set_ylabel("Secrecy Capacity Variance")
var_plt.set_xlabel("System Size")

ave_plt.set_ylabel("Secrecy Capacity Average (Bits/Symbol)")
ave_plt.set_xlabel("System Size")

ave_plt.grid(True, which='both')
seaborn.despine(ax=ave_plt, offset=0)
ave_plt.legend(loc="upper right")

var_plt.grid(True, which='both')
seaborn.despine(ax=var_plt, offset=0)
var_plt.legend(loc="upper right")
plt.show()


