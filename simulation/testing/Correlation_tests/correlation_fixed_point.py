# A test for looking at impact of adding correlation through concatenated channels on the channel capacity
# Takeaway:
# If power is allocated equally the ave capacity doesn't seem to be affected however variance increase
# with increasing correlation. However if water-filling
# is used then the correlated channels seem to have increasing capacity as correlation increases

import matplotlib.pyplot as plt
from src.lin_alg import *
from src.stieltjes_eqn import *
import seaborn
from scipy.linalg import toeplitz
from src.circle_laws import *
from scipy.optimize import *
from scipy.optimize import fsolve

rows = 100
cols = 100
H_cor = c_rand(rows, cols)
block = toeplitz(np.array((1, .5)))
correlation_matrix = block_matrix(block, H_cor.shape[0], normalize=True)
correlation_matrix = np.sqrt(correlation_matrix)
e_val_correlation = np.linalg.eigvalsh(correlation_matrix)
# H_cor = correlation_matrix@H_cor
HH = H_cor@hermetian(H_cor)
e_val = np.linalg.eigvalsh(HH)
AED, bins = np.histogram(np.asarray(e_val), bins=50)
x_values, step = np.linspace(.01, 10, 100, retstep=True)
s_values = x_values + 1j*1e-6

def rayleigh_steiltjes(z, gamma_inv_z):
    total = - (1/2)*np.sqrt((1-4*gamma_inv_z)) - 1/2
    total /= (1+z)
    if np.isnan(total) or np.isinf(total):
        print("nan")
    return total

def gamma_c_fixed(z, gamma_inv_z):
    """
    Using an out of scope variable here because I don't want to pass it though another fixed point equation.
    :param s:
    :param gamma_s:
    :return:
    """
    num = e_val_correlation.size
    total = 0
    for eigen_val in e_val_correlation:
        total -= (1/num)*(1/(eigen_val-1/gamma_inv_z))
    total /= (1+z)
    if np.isnan(total) or np.isinf(total):
        print("nan")
    return total

def S_ch(z):
    # check = fixed_point(gamma_c_fixed, z, 100, tolerance=1e-3)/z
    #My Test
    # check = fixed_point(rayleigh_steiltjes, z, 100, tolerance=1e-3)/z
    # check = 1/np.power((1+z), 2)
    check = 1/(1+z)
    #TODO current problem is that gamma_s is zero
    if np.isnan(check) or np.isinf(check):
        print("catch")
    return check

def gammafixed(z, gamma_z):
    check = S_ch(gamma_z)
    if np.isinf(check) or np.isnan(check):
        print("inf/nan")
    # return np.power(gamma_z/z, 1 / 2) - 1
    # return np.power(gamma_z / z, 1 / 4) - 1
    return (check*gamma_z)/z-1
    # return z*(1+gamma_z)/check

def f_solve_fun(gamma_z, z_val):
    # return (S_ch(gamma_z)*gamma_z)/z_val-1
    return z_val*(1+gamma_z)/S_ch(gamma_z)


def built_in_fixed(values, function):
    for val in values:
        init = np.random.uniform(0, 1) + 1j * np.random.uniform(0, 1)
        attempt = root(f_solve_fun, init, args=val)
        print(attempt)
    init = np.random.uniform(0, 1, values.shape) + 1j * np.random.uniform(0, 1, values.shape)
    attempt = root(function, init, args=values)
    return attempt.x

stieltjes_values = steiltjes_from_gamma(s_values, gammafixed)
pdf = estimated_pdf(stieltjes_values)
print(f"asymptotic capacity: {aed_capacity(x_values, pdf, 1/rows, rows, step=step)}")
print(f"true capacity: {capacity(HH, 1/rows)}")

fig, ax = plt.subplots()
plt.title("IRS Channels Components: i.i.d, $\mathbb{N}(0,1/N), H = H_1 \Phi H_2$")
ax.plot(x_values, pdf, label='asymptotic')
ax.plot(bins[:-1], AED/rows, label='total: with IRS')
plt.legend(loc="upper right")
ax.grid(True, which='both')
seaborn.despine(ax=ax, offset=0)
plt.show()


