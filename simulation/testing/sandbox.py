import matplotlib.pyplot as plt
from src.fixed_point import *
from src.lin_alg import *
from src.network_simulation_tools import water_filling
import seaborn
from scipy.linalg import toeplitz
from scipy.optimize import fixed_point, root
from scipy.optimize import fsolve

capacities = []
average = 10000
for i in range(average):
    check = pow(np.abs(np.random.randn(1)), 2) + 1
    # capacities.append(check)
    capacities.append(np.log2(check))
print(np.average(capacities))
