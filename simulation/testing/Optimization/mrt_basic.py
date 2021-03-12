from communication_util.beamformers import maximum_ratio_transmission
from communication_util.detectors import ASK_2_Slicer
from src.lin_alg import c_rand
import matplotlib.pyplot as plt
import numpy as np


# BER evaluation framework
num_simulations = 100
per_channel_symbols_num = 1000
SNRs_dB = np.linspace(-10, 20, 10)
SNRs = np.power(10, SNRs_dB/10)
num_rx = 1
tx_num_list = [1, 2, 4]
num_tx = 8
alphabet = np.round(np.linspace(-1, +1, 2))
fig = plt.figure(1)
for num_tx in tx_num_list:
    BER_list = []
    for snr in SNRs:
        errors = 0
        for simulation_ind in range(num_simulations):
            los_channel = np.random.standard_normal((1, num_tx))/np.sqrt(2) + 1j*np.random.standard_normal((1, num_tx))/np.sqrt(2)
            beamformer = maximum_ratio_transmission(los_channel.T)
            for i in range(per_channel_symbols_num):
                symbol = alphabet[np.random.randint(0, 2)]
                # transmit_symbol = symbol*beamformer/np.sqrt(num_tx)
                transmit_symbol = symbol*beamformer
                noise = np.random.standard_normal()*np.sqrt((1 / (snr)))
                received_signal = los_channel@transmit_symbol + noise
                detected_symbol = ASK_2_Slicer(received_signal)
                errors += symbol != detected_symbol
                pass
        BER_list.append(errors/(num_simulations*per_channel_symbols_num))   # only for N_r = 1
    plt.plot(SNRs_dB, BER_list, label=f"MRT LOS # Tx: {num_tx}", linestyle='--', marker='o')
plt.xlabel(r'10log$(E[x]/\sigma^2_n$) [dB]')
plt.ylabel("BER")
plt.xscale('linear')
plt.yscale('log')
title = "MRT + LOS "
plt.title(title)
plt.grid(True)
plt.legend(loc='lower left')
plt.show()
time_path = "Output/MRT_vs_Tx_not_normalized.png"
fig.savefig(time_path, format="png")