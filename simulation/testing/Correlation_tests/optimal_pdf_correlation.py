
import matplotlib.pyplot as plt
from src.lin_alg import *
from src.network_simulation_tools import water_filling

size = 100
bins = int(size/2)
irs_cor = []
rank = 100
phi = rank/size
average = 1
correlations = [.5, .8, .9, .98]
for cor_rho in correlations:
    capacity_variance = []
    capacity_average = []
    t_correlation_matrix = exponential_correlation(size, cor_rho)
    s_correlation_matrix = exponential_correlation(size, cor_rho)
    r_correlation_matrix = exponential_correlation(size, cor_rho)
    powers = []
    for i in range(average):
        G = c_rand(size, size)
        G2 = c_rand(size, size)
        H_cor = r_correlation_matrix@G2@s_correlation_matrix@G@t_correlation_matrix
        powers.append(water_filling(H_cor, 1, vals=True))
    density, power_bin = np.histogram(np.asarray(powers), bins=bins)
    plt.plot(power_bin[:-1], density/average, '-o', label='power allocation')
plt.legend()
plt.show()


