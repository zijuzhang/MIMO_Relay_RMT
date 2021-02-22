import numpy as np
from src.lin_alg import *
from src.optimization import *
import matplotlib.pyplot as plt
average = 10
num_tx = 8
num_rx = 8
num_optimization_rounds = 6
IRS_sizes = [4, 8, 16, 64, 128]
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
        H_t = F2@F1
        rand = random_phase(num_reflectors)
        beamformer = precode_mf(F2@F1)
        coefficients = []
        non_cross = -1j*1j*0
        for elem_ind in range(num_reflectors):
            #   Pick out rows and columns of the channels corresponding to specific IRS elements
            f2i = F2[:, elem_ind]
            f1i = F1[elem_ind, :]
            i_mat = np.outer(f2i, f1i)
            non_cross += np.trace(i_mat@beamformer@hermetian(i_mat@beamformer))
            # Add all of the cross components of the polynomial to the list of coefficients.
            for elem2_ind in range(num_reflectors - elem_ind - 1):
                elem2_ind += elem_ind + 1
                f2j = F2[:, elem2_ind]
                f1j = F1[elem2_ind, :]
                j_mat = np.outer(f2j, f1j)
                # Below wraps around to get complex scalar by the trace rotation property.
                coefficient = f1i.T@beamformer@hermetian(beamformer)@np.conjugate(f1j)*hermetian(f2j)@f2i
                # coefficient = f1i.T@np.conjugate(f1j)*hermetian(f2j)@f2i
                # coefficient1 = np.trace(i_mat@transmit_matched_rand@hermetian(transmit_matched_rand)@hermetian(j_mat))
                coefficients.append(coefficient)
        # Create matrix to pick out the MSE terms influenced by specific IRS elements.
        phase_adjacency = np.concatenate((np.zeros((num_reflectors, len(coefficients))), np.eye(num_reflectors)), axis=1)
        # correct = np.allclose(Fc, F2@F1)
        #TODO check if this should actually be 0j
        extra = 1j*-1j
        for elem_ind in range(num_reflectors):
            f2i = F2[:, elem_ind]
            f1i = F1[elem_ind, :]
            # Is this coming from the MMSE term?
            coefficients.append(-np.trace(np.outer(f2i, f1i)@beamformer))
            # Non-cross terms are not important for optimization but keep track of the total term.
            extra += np.trace(np.outer(f2i, f1i)@beamformer@hermetian(np.outer(f2i, f1i)@beamformer))
        ind = 0
        for elem_ind in range(num_reflectors-1):
            for elem2_ind in range(num_reflectors - elem_ind - 1):
                phase_adjacency[elem_ind, ind + elem2_ind] = 1
                elem3_ind = elem2_ind + elem_ind + 1
                phase_adjacency[elem3_ind, ind + elem2_ind] = 1
            ind += num_reflectors - elem_ind - 1
        for round in range(num_optimization_rounds):
            coefficients = get_polynomial_coefficients(F1, F2, beamformer, num_reflectors)
            optimal_phases, improvements = optimize_phases_preselected(np.asarray(coefficients),
                                                           phase_adjacency, F2, F1, csi=1, num_repetitions=2,
                                                           improvements=True)
            H_opt = F2@optimal_phases@F1
            beamformer = precode_mf(H_opt)
        check_1 = np.trace(-H_t@beamformer) + np.conjugate(np.trace(-H_t@beamformer)) + np.trace(H_t@beamformer@hermetian(H_t@beamformer))
        check_2 = np.trace(-H_opt@beamformer) + np.conjugate(np.trace(-H_opt@beamformer)) + np.trace(H_opt@beamformer@hermetian(H_opt@beamformer))
        #   Now perform comparison of optimized phases with random/uniform phases
        # transmit_symbols = BPSK(num_rx)
        transmit_averge = 10
        trasmit_ave_list_rand = []
        trasmit_ave_list_opt = []
        for i in range(transmit_averge):
            noise = c_rand(num_rx, 1, var=1/num_tx).flatten() # Gives SNR of 0 dB
            # noise = c_rand(num_rx, 1).flatten()
            transmit_symbols = c_rand(num_tx, 1, var=1).flatten()
            received_rand = H_t@beamformer@transmit_symbols + noise
            check1 = np.power(np.linalg.norm(received_rand - transmit_symbols), 2)
            trasmit_ave_list_rand.append(np.power(np.linalg.norm(received_rand - transmit_symbols), 2))
        #   Line below could be done by going through each element and shifting the phase. Line below is faster.
        #     transmit_matched_opt = precode_mf(F2@optimal_phases@F1)
            received_opt = F2@optimal_phases@F1@beamformer@transmit_symbols + noise
            check2 = np.power(np.linalg.norm(received_opt - transmit_symbols), 2)
            trasmit_ave_list_opt.append(np.power(np.linalg.norm(received_opt-transmit_symbols), 2))
        MSE_rand.append(10 * np.log(np.average(trasmit_ave_list_rand)))
        MSE_optimized.append(10 * np.log(np.average(trasmit_ave_list_opt)))
    MSE_opt_size.append(np.average(MSE_optimized))
    MSE_rand_size.append(np.average(MSE_rand))
fig = plt.figure()
ave_plt = fig.add_subplot(1, 1, 1)
ave_plt.set_ylabel("Average MSE  (dB)")
ave_plt.set_xlabel("Number of IRS elements")
ave_plt.set_yscale('log')
ave_plt.plot(IRS_sizes, MSE_opt_size, '-o', label='MSE: optimized phases')
ave_plt.plot(IRS_sizes, MSE_rand_size, '-o', label='MSE: Random phases')
ave_plt.legend(loc="upper right")
ave_plt.set_title(f'MSE  vs Number of IRS elements with N_T = {num_tx}, N_R = {num_rx} with SNR ='
                  f' {10 * np.log(1)} dB')
plt.show()