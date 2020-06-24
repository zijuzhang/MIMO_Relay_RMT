import matplotlib.pyplot as plt
from src.correlated_capacities import *
snrs = [5, 10, 20]
system_sizes = [5, 20, 50, 100]
correlations = [.5, .8, .98]
fig = plt.figure()
loyka = fig.add_subplot(1, 1, 1)
snr = pow((20/10), 10)
for system_size in system_sizes:
    loyka_capacity = []
    for correlation in correlations:
        loyka_capacity.append(loyka_upper_bound(system_size, correlation, snr))
    loyka.plot(correlations, loyka_capacity, '-o', label=f'SNR: {system_size}')
plt.show()