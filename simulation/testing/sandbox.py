import matplotlib.pyplot as plt
from src.fixed_point import *
from src.lin_alg import *
from src.network_simulation_tools import water_filling
import seaborn
from scipy.linalg import toeplitz


one = np.eye(4)*2
two = np.eye(4)*1
# two[:,0] = 4
check1 = np.linalg.eig(one)
check2 = np.linalg.eig(two)
print("done")


