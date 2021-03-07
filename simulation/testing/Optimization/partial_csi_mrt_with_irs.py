from communication_util.beamformers import maximum_ratio_transmission
from communication_util.detectors import ASK_2_Slicer
from src.lin_alg import c_rand
import matplotlib.pyplot as plt
import numpy as np


# BER evaluation framework
BER_list = []
num_simulations = 100
per_channel_symbols_num = 100
SNRs_dB = np.linspace(-10, 5, 10)
SNRs = np.power(10, SNRs_dB/10)
num_rx = 1
num_surfaces = 8
num_tx = 8
alphabet = np.round(np.linspace(-1, +1, 2))
plt.figure(1)
for snr in SNRs:
    errors = 0
    for simulation_ind in range(num_simulations):
        los_channel = np.random.standard_normal((1, num_tx)) + 1j*np.random.standard_normal((1, num_tx))
        H1 = c_rand(num_surfaces, num_tx)
        h2 = np.random.standard_normal((num_surfaces, 1))/np.sqrt(2) + 1j*np.random.standard_normal((num_surfaces, 1))/np.sqrt(2)
        beamformer = maximum_ratio_transmission(los_channel.T, partial=0)
        # beamformer = maximum_ratio_transmission((h2.T@H1).T, partial=0)
        # beamformer = maximum_ratio_transmission((h2.T@H1).T)
        for i in range(per_channel_symbols_num):
                symbol = alphabet[np.random.randint(0, 2)]
                transmit_symbol = symbol*beamformer/np.sqrt(num_tx)
                # noise = np.random.standard_normal()*np.sqrt((1 / (snr*2))) + \
                #     1j*np.random.standard_normal()*np.sqrt((1 / (snr*2)))
                noise = np.random.standard_normal()*np.sqrt((1 / (snr)))
                received_signal = los_channel@transmit_symbol + noise
                # received_signal = h2.T@H1@transmit_symbol + noise
                detected_symbol = ASK_2_Slicer(received_signal)
                errors += symbol != detected_symbol
    BER_list.append(errors/(num_simulations*per_channel_symbols_num))   # only for N_r = 1
plt.xlabel(r'10log$(E[x]/\sigma^2_n$) [dB]')
plt.ylabel("SER")
plt.xscale('linear')
plt.yscale('log')
plt.grid(True)
plt.legend(loc='lower left')
plt.plot(SNRs_dB, BER_list, linestyle='--', marker='o')
plt.show()