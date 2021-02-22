# A test for looking at impact of adding correlation through concatenated channels on the channel capacity
# Takeaway:
# If power is allocated equally the ave capacity doesn't seem to be affected however variance increase
# with increasing correlation. However if water-filling
# is used then the correlated channels seem to have increasing capacity as correlation increases

import matplotlib.pyplot as plt
from src.lin_alg import *
from src.optimization import *
from src.network_simulation_tools import water_filling
import seaborn

fig = plt.figure()
ave_plt = fig.add_subplot(2, 1, 1)
var_plt = fig.add_subplot(2, 1, 2)
num_channel_average = 100
num_symbol_average = 100
num_opt_rounds = 3
IRS_sizes = [16, 32]
num_rx = 8
num_tx = 8

correlations = np.linspace(0, 1-10e-6, 5)   # Don't want to go all the way to 1, just close.
# Evaluate over different correlations.
for num_reflectors in IRS_sizes:
    MSE_variance = []
    MSE_average = []
    MSE_variance_opt = []
    MSE_average_opt = []
    capacity_average = []
    capacity_variance = []
    capacity_average_opt = []
    capacity_variance_opt = []
    # For each correlation level, evaluate over different system sizes.
    for cor_rho in correlations:
        t_correlation_matrix = exponential_correlation(num_tx, cor_rho)
        r_correlation_matrix = exponential_correlation(num_rx, cor_rho)
        cor_cap = []
        cor_cap_opt = []
        for i in range(num_channel_average): #  Average over channel realizations
            F1 = c_rand(num_reflectors, num_tx, var=1)
            F2 = c_rand(num_rx, num_reflectors, var=1)
            H_t = F2@F1
            H_cor = r_correlation_matrix@F2@random_phase(num_reflectors)@F1@t_correlation_matrix
            # Initial beamformer -> can be updated in subsequent iterations.
            beamformer = precode_mf(H_cor)
            # Loop for iterating between optimizing MF then phases.
            # Because phase optimization is just an improvement, reiterate (need to make sure this is in a different
            # order each time!!?)
            coefficients = get_polynomial_coefficients(F1@t_correlation_matrix, r_correlation_matrix@F2, beamformer, num_reflectors)
            phase_adjacency = get_polynomial_adjacency_matrix(num_reflectors, len(coefficients) - num_reflectors)
            optimal_phases = 0
            for j in range(num_opt_rounds):
                # Coefficients passed to the optimization are based on the beamformer.
                coefficients = get_polynomial_coefficients(F1@t_correlation_matrix, r_correlation_matrix@F2, beamformer, num_reflectors)
                optimal_phases, improvements = optimize_phases_preselected(np.asarray(coefficients),
                                                                           phase_adjacency, F2, F1, csi=1, num_repetitions=2,
                                                                           improvements=True, maximize=True)
                H_cor_current = r_correlation_matrix@F2@optimal_phases@F1@t_correlation_matrix
                beamformer = precode_mf(H_cor_current)

            H_cor_opt = r_correlation_matrix@F2@optimal_phases@F1@t_correlation_matrix
            H_cor = r_correlation_matrix@F2@random_phase(num_reflectors)@F1@t_correlation_matrix
            # Initial beamformer -> can be updated in subsequent iterations.
            HH_cor = H_cor@hermetian(H_cor)
            HH_cor_opt = H_cor_opt@hermetian(H_cor_opt)
            # Now that phases and beamformer are optimized, evaluate an average MSE.
            # capacity_list_rand = []
            # capacity_list_opt = []
            # for k in range(num_symbol_average):
                # Generate instance of noise and symbols
                # noise = c_rand(num_rx, 1).flatten()
                # transmit_symbols = c_rand(size, 1).flatten()
            cor_cap.append(capacity(HH_cor, 1/num_tx))
            cor_cap_opt.append(capacity(HH_cor_opt, 1/num_tx))
            # cor_cap.append(capacity_water_filled(water_filling(H_cor, 1)))
        capacity_average.append(np.average(cor_cap))
        capacity_variance.append(np.var(cor_cap))
        capacity_average_opt.append(np.average(cor_cap_opt))
        capacity_variance_opt.append(np.var(cor_cap_opt))

    var_plt.plot(correlations, capacity_variance, '-o', label=f'Tx=Tr=: {num_reflectors}')
    ave_plt.plot(correlations, capacity_average, '-o', label=f'Tx=Tr=: {num_reflectors}')
    var_plt.plot(correlations, capacity_variance_opt, '-o', label=f'Opt Tx=Tr=: {num_reflectors}')
    ave_plt.plot(correlations, capacity_average_opt, '-o', label=f'Opt Tx=Tr=: {num_reflectors}')
# ave_plt.title("Capacity Average")
# plt.title("Capacity Variance")
var_plt.set_ylabel("Capacity Variance")
var_plt.set_xlabel("System Size")

ave_plt.set_ylabel("Capacity Average (Bits/Channel Use)")
ave_plt.set_xlabel("Correlation Level (rho)")

ave_plt.grid(True, which='both')
seaborn.despine(ax=ave_plt, offset=0)
ave_plt.legend(loc="upper right")

var_plt.grid(True, which='both')
seaborn.despine(ax=var_plt, offset=0)
var_plt.legend(loc="upper right")
plt.show()


