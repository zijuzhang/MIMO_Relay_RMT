import numpy as np
from src.lin_alg import *
from src.optimization import *
import matplotlib.pyplot as plt 

average = 200
num_tx = 10
num_rx = 10
IRS_sizes = [10, 16, 64]
MSE_opt_size = []
MSE_rand_size = []
for size in IRS_sizes:
    MSE_rand = []
    MSE_optimized = []
    num_reflectors = size
    for j in range(average):
        #   Each IRS path is a column of F2 @ a row of F1
        F1 = c_rand(num_reflectors, num_tx, var=1)
        F2 = c_rand(num_rx, num_reflectors, var=1)
        rand = random_phase(num_reflectors)
        transmit_matched_rand = precode_mf(F2@rand@F1)
        coefficients = []
        #   Choose the optimal IRS phases using the known channels with 0 phase at the IRS elements.
        for elem_ind in range(num_reflectors):
            ind = elem_ind * num_reflectors
            for elem2_ind in range(num_reflectors - elem_ind - 1):
                elem2_ind += elem_ind + 1
                f2i = F2[:, elem_ind]
                f1i = F1[elem_ind, :]
                f2j = F2[:, elem2_ind]
                f1j = F1[elem2_ind, :]
                # coefficient = f1i.T@transmit_matched_rand@hermetian(transmit_matched_rand)@np.conjugate(f1j)*hermetian(f2j)@f2i
                coefficient = f1i.T@np.conjugate(f1j)*hermetian(f2j)@f2i
                coefficients.append(coefficient)
        phase_adjacency = np.zeros((num_reflectors, len(coefficients)))
        ind = 0
        for elem_ind in range(num_reflectors-1):
            for elem2_ind in range(num_reflectors - elem_ind - 1):
                phase_adjacency[elem_ind, ind + elem2_ind] = 1
                elem3_ind = elem2_ind + elem_ind + 1
                phase_adjacency[elem3_ind, ind + elem2_ind] = 1
                pass
            ind += num_reflectors - elem_ind - 1
        # optimal_phases = optimize_phases(np.asarray(coefficients), phase_adjacency, 1)
        #TODO toggle negative
        optimal_phases = optimize_phases(np.asarray(coefficients), phase_adjacency, 2)
        check1 = np.trace(F2@rand@F1@hermetian(F2@rand@F1))
        check2 = np.trace(F2 @ np.diag(optimal_phases) @ F1 @ hermetian(F2 @ np.diag(optimal_phases) @ F1))
        #   Now perform comparison of optimized phases with random/uniform phases
        # transmit_symbols = BPSK(num_rx)
        transmit_symbols = c_rand(num_tx, 1, var=1).flatten()
        received_rand = F2@rand@F1@transmit_matched_rand@transmit_symbols
        transmit_matched_opt = precode_mf(F2@np.diag(optimal_phases)@F1)
        received_opt = F2@np.diag(optimal_phases)@F1@transmit_matched_opt@transmit_symbols
        MSE_rand.append(10 * np.log(np.power(np.linalg.norm(received_rand - transmit_symbols), 2)))
        MSE_optimized.append(10 * np.log(np.power(np.linalg.norm(received_opt - transmit_symbols), 2)))
    MSE_opt_size.append(np.average(MSE_optimized))
    MSE_rand_size.append(np.average(MSE_rand))
    
fig = plt.figure()
ave_plt = fig.add_subplot(2, 1, 1)
var_plt = fig.add_subplot(2, 1, 2)
ave_plt.set_ylabel("Average MSE  (dB)")
ave_plt.set_yscale('log')
ave_plt.legend(loc="upper left")
ave_plt.plot(IRS_sizes, MSE_opt_size, '-o', label='opt MSE MF')
ave_plt.plot(IRS_sizes, MSE_rand_size, '-o', label='rand MSE MF')
ave_plt.legend(loc="upper left")
plt.show()