from communication_util.beamformers import maximum_ratio_transmission
from communication_util.detectors import ASK_2_Slicer
from src.optimization import optimize_MISO_phase
from src.lin_alg import c_rand
import matplotlib.pyplot as plt
import numpy as np
import time


# BER evaluation framework

num_simulations = 100
per_channel_symbols_num = 1000
SNRs_dB = np.linspace(-10, 15, 8)
SNRs = np.power(10, SNRs_dB/10)
num_rx = 1
num_surfaces = 1
num_tx = 2
alphabet = np.round(np.linspace(-1, +1, 2))
IRS_partial = 1
MRT_partial = 1
fig = plt.figure(1)
BER_list = []
BER_list_opt = []
for snr in SNRs:
    errors = 0
    errors_opt = 0
    for simulation_ind in range(num_simulations):
        los_channel = np.random.standard_normal((1, num_tx)) + 1j*np.random.standard_normal((1, num_tx))
        H1 = c_rand(num_surfaces, num_tx)
        h2 = np.random.standard_normal((num_surfaces, 1))/np.sqrt(2) + 1j*np.random.standard_normal((num_surfaces, 1))/np.sqrt(2)
        beamformer_mrt = maximum_ratio_transmission(los_channel.T)
        beamformer_irs = maximum_ratio_transmission((h2.T@H1+los_channel).T)
        for i in range(per_channel_symbols_num):
            symbol = alphabet[np.random.randint(0, 2)]
            noise = np.random.standard_normal()*np.sqrt((1 / (snr)))
            received_signal = los_channel@ \
                              (symbol*beamformer_mrt/np.sqrt(num_tx)) + noise
            received_signal_opt = (h2.T@H1+los_channel)@ \
                                  (symbol*beamformer_irs/np.sqrt(num_tx)) + noise
            detected_symbol = ASK_2_Slicer(received_signal)
            errors += symbol != detected_symbol
            detected_symbol = ASK_2_Slicer(received_signal_opt)
            errors_opt += symbol != detected_symbol
    BER_list.append(errors/(num_simulations*per_channel_symbols_num))   # only for N_r = 1
    BER_list_opt.append(errors_opt/(num_simulations*per_channel_symbols_num))   # only for N_r = 1
plt.plot(SNRs_dB, BER_list, label=f"MRT with no IRS path", linestyle='--', marker='o')
plt.plot(SNRs_dB, BER_list_opt, label=f"MRT with IRS path", linestyle='--', marker='x')
plt.xlabel(r'10log$(E[x]/\sigma^2_n$) [dB]')
plt.ylabel("BER")
plt.xscale('linear')
plt.yscale('log')
title = "IRS + LOS "
plt.title(title)
plt.grid(True)
plt.legend(loc='lower left')
# plt.show()
time_path = f"Output/MRT_With_Without_IRSPath+" + f"{time.time()}" + ".png"
fig.savefig(time_path, format="png")