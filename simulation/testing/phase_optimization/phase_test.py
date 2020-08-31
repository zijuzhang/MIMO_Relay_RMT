# A test for looking at impact of adding correlation through concatenated channels on the channel capacity
# Takeaway:
# If power is allocated equally the ave capacity doesn't seem to be affected however variance increase
# with increasing correlation. However if water-filling
# is used then the correlated channels seem to have increasing capacity as correlation increases

import matplotlib.pyplot as plt
from src.fixed_point import *
from src.lin_alg import *
from src.network_simulation_tools import water_filling
average = 100
sizes = [10, 100]
capacity_variance = []
capacity_average = []
for size in sizes:
    G = c_rand(size, size)
    G2 = c_rand(size, size)
    for i in range(average):
        G2 *= np.diag(np.exp(1j * np.random.uniform(0, 2 * np.pi, 1)))
        G *= np.diag(np.exp(1j * np.random.uniform(0, 2 * np.pi, 1)))
        main = np.trace(G@hermetian(G) + G2@hermetian(G2))
        cross = np.trace(G@np.conjugate(G2) + G2@np.conjugate(G))
        capacity_average.append(main)
        capacity_variance.append(cross)
    pass
pass
