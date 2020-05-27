import matplotlib.pyplot as plt
from src.fixed_point import *
from src.lin_alg import *
from src.network_simulation_tools import water_filling
import seaborn
from scipy.linalg import toeplitz
from scipy.optimize import fixed_point, root
from scipy.optimize import fsolve



one = np.eye(4)*2
two = np.eye(4)*1
# two[:,0] = 4
check1 = np.linalg.eig(one)
check2 = np.linalg.eig(two)
print("done")




def f_solve_check(gamma_z):
    return 1/7*(pow(gamma_z, 3)+2)

func = lambda x : np.power(x, 3) - 7*x +2
x_values, step = np.linspace(.01, 10, 100, retstep=True)
check = root(func, .1)
val = check.x
plt.plot(x_values, func(x_values))
plt.scatter(val, func(val))
plt.show()


func = lambda x: (1+1j)*np.power(x, 3) - (2j+7)*x +2
check = root(func)
