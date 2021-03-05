from communication_util.beamformers import *
from communication_util.detectors import ASK_2_Slicer


# BER evaluation framework
BER_list = []
num_simulations = 1000
per_channel_symbols_num = 1000
SNRs_dB = np.linspace(0, 30, 2)
SNRs = np.power(10, SNRs_dB/10)
num_rx = 1
num_surfaces = 0
num_tx = 8
alphabet = np.linspace(-1, +1, 2)
for snr in SNRs:
    errors = 0
    for simulation_ind in range(num_simulations):
        # TODO add option for MISO (all have same symbol)
        channel = np.random.standard_normal((num_tx, 1))
        beamformer = maximum_ratio_transmission(channel)
        for i in range(per_channel_symbols_num):
            symbols = alphabet[np.random.random_integers(0, 1, (1, 1))] # Adjust power for MISO case.
            noise = np.random.standard_normal((num_rx, 1))*np.sqrt((1 / (snr*2))) + \
                1j*np.random.standard_normal((num_rx, 1))*np.sqrt((1 / (snr*2)))
            received_signal = symbols*channel + noise
            detected_symbols = ASK_2_Slicer(received_signal)
            errors += received_signal == detected_symbols
    BER_list.append(errors/(num_simulations*per_channel_symbols_num))   # only for N_r = 1
    pass
            # Setup MISO channel with Maximum Ratio Transmission
            # first generate a channel realization (start by using

            # Simulation BER of this channel realization.