import numpy as np
from src.lin_alg import *
from src.stieltjes_eqn import estimated_pdf
import matplotlib.pyplot as plt
from src.fixed_point import *
from scipy.linalg import toeplitz


def deterministic_stieltjes(s_vals):
    num = e_val_correlation.size
    if type(s_vals) is not np.ndarray:
        total = 1j * 0
    else:
        total = 1j*np.zeros(s_vals.shape)
    for e_val in e_val_correlation:
        total += 1/((e_val-s_vals)*num)
    return total

def inv_gamme_fixed(z, inv_gamma_z):
    return - deterministic_stieltjes(1/inv_gamma_z)/(z+1)

def inverse_gamma(z_vector):
    check = fixed_point(inv_gamme_fixed, z_vector, 100, unique_half_plane=True)
    return check

def s_transform(z):
    check = ((1+z)/z)*inverse_gamma(z)
    return check

def gamma_fixed(z, gamma_z):
    # check = (gamma_z + 1)*z/s_transform(z)
    check = gamma_z*s_transform(z) - z
    return check

def gamma(z):
    check = fixed_point(gamma_fixed, z, 100)
    return check

def steiltjes_from_gamma(s):
    check = (-1/s)*(1+gamma(1/s))
    return check

#   First Generate a deterministic matric
cols = 100
rows = 100
block = toeplitz(np.array((1, .5)))
correlation_matrix = block_matrix(block, rows, normalize=True)
e_val_correlation = np.linalg.eigvalsh(correlation_matrix)

x_vals = np.linspace(1e-6, 3, 300)
# x_vals  = e_val_correlation
s_vals = x_vals + 1j*1e-6
pdf = estimated_pdf(deterministic_stieltjes(s_vals))
fixed_pdf = estimated_pdf(steiltjes_from_gamma(s_vals))
irs_AED, bins_irs = np.histogram(e_val_correlation, bins=50)

fig = plt.figure()
first = fig.add_subplot(3, 1, 1)
second = fig.add_subplot(3, 1, 2)
third = fig.add_subplot(3, 1, 3)

first.stem(x_vals, pdf/np.linalg.norm(pdf), label=f"estimate")
second.stem(x_vals, fixed_pdf/np.linalg.norm(fixed_pdf), label=f"fixed_point")
third.stem(bins_irs[:-1], irs_AED/100, label='True')
plt.legend(loc="upper right")
# ax.grid(True, which='both')
# seaborn.despine(ax=ax, offset=0)  # the important part here
plt.show()