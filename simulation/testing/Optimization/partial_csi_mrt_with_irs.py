from communication_util.beamformers import maximum_ratio_transmission
from communication_util.detectors import ASK_2_Slicer
from src.optimization import optimize_MISO_phase
from src.lin_alg import c_rand
import matplotlib.pyplot as plt
import numpy as np


# BER evaluation framework
BER_list = []
BER_list_opt = []
num_simulations = 100
per_channel_symbols_num = 1000
SNRs_dB = np.linspace(-10, 5, 10)
SNRs = np.power(10, SNRs_dB/10)
num_rx = 1
num_surfaces = 16
num_tx = 8
alphabet = np.round(np.linspace(-1, +1, 2))
CSI_partial = 1
plt.figure(1)
for snr in SNRs:
    errors = 0
    errors_opt = 0
    for simulation_ind in range(num_simulations):
        los_channel = np.random.standard_normal((1, num_tx)) + 1j*np.random.standard_normal((1, num_tx))
        H1 = c_rand(num_surfaces, num_tx)
        h2 = np.random.standard_normal((num_surfaces, 1))/np.sqrt(2) + 1j*np.random.standard_normal((num_surfaces, 1))/np.sqrt(2)
        phase = optimize_MISO_phase(np.sum(los_channel), h2.T*np.sum(H1, 1))
        # beamformer = maximum_ratio_transmission(los_channel.T, partial=1)
        # beamformer = maximum_ratio_transmission((h2.T@H1).T, partial=1)
        # beamformer = maximum_ratio_transmission((h2.T@H1+los_channel).T, partial=1)
        beamformer = maximum_ratio_transmission((h2.T@np.diag(phase)@H1+los_channel).T, partial=1)
        for i in range(per_channel_symbols_num):
                symbol = alphabet[np.random.randint(0, 2)]
                transmit_symbol = symbol*beamformer/np.sqrt(num_tx)
                # noise = np.random.standard_normal()*np.sqrt((1 / (snr*2))) + \
                #     1j*np.random.standard_normal()*np.sqrt((1 / (snr*2)))
                noise = np.random.standard_normal()*np.sqrt((1 / (snr)))
                # received_signal = los_channel@transmit_symbol + noise
                # received_signal = h2.T@H1@transmit_symbol + noise
                received_signal = (h2.T@np.eye(num_surfaces)@H1+los_channel)@transmit_symbol + noise
                received_signal_opt = (h2.T@np.diag(phase)@H1+los_channel)@transmit_symbol + noise
                detected_symbol = ASK_2_Slicer(received_signal)
                errors += symbol != detected_symbol
                detected_symbol = ASK_2_Slicer(received_signal_opt)
                errors_opt += symbol != detected_symbol
    BER_list.append(errors/(num_simulations*per_channel_symbols_num))   # only for N_r = 1
    BER_list_opt.append(errors_opt/(num_simulations*per_channel_symbols_num))   # only for N_r = 1
plt.xlabel(r'10log$(E[x]/\sigma^2_n$) [dB]')
plt.ylabel("BER")
plt.xscale('linear')
plt.yscale('log')
title = "LOS with" + f"CSI: {CSI_partial}"
# title = "IRS with" + f"CSI: {CSI_partial}"
# title = "IRS + LOS  with" + f"CSI: {CSI_partial}"
plt.title(title)
plt.grid(True)
plt.plot(SNRs_dB, BER_list, label='Zero Phase', linestyle='--', marker='o')
plt.plot(SNRs_dB, BER_list_opt,label='Optimized IRS', linestyle='--', marker='x')
plt.legend(loc='lower left')
plt.show()
time_path = "Output/IRS_LOS_BER_"+f"CSI_{CSI_partial}" +".png"
plt.savefig(time_path, format="png")