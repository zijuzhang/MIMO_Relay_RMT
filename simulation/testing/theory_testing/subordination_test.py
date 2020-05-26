import numpy as np
from src.lin_alg import *
from src.stieltjes_eqn import *
import matplotlib.pyplot as plt
from src.fixed_point import *
from scipy.linalg import toeplitz



#   First Generate a deterministic matric
cols = 100
rows = 100
block = toeplitz(np.array((1, .5)))
correlation_matrix = block_matrix(block, rows, normalize=True)
correlation_matrix = correlation_matrix@hermetian(correlation_matrix)
e_val_correlation = np.linalg.eigvalsh(correlation_matrix)

x_vals, step = np.linspace(1e-6, 3, 100, retstep=True)
# x_vals = np.unique(e_val_correlation) #    Need to get rid of duplicates here
s_vals = x_vals + 1j*1e-6
pdf = normalize_pdf(estimated_pdf(aed_from_svd(s_vals, g_tilde))) #   Is Normalizing this reasonable?

check = estimated_pdf(deterministic_stieltjes(s_vals))
# pdf = estimated_pdf(deterministic_stieltjes(s_vals)) #   Is Normalizing this reasonable?
pdf = normalize_pdf(estimated_pdf(deterministic_stieltjes(s_vals))) #   Is Normalizing this reasonable?
# fixed_pdf = estimated_pdf(steiltjes_from_gamma(s_vals, gamma))
theoretic_capacity = aed_capacity(x_vals, pdf, 1/cols, rows)    #Shouldn't be using step here? try without normaling but with step
irs_AED, bins_irs = np.histogram(e_val_correlation, bins=50)

print(f"simulated capacity: {capacity(correlation_matrix, 1/cols)}")
print(f"theoretic capacity: {theoretic_capacity}")

fig = plt.figure()
first = fig.add_subplot(3, 1, 1)
second = fig.add_subplot(3, 1, 2)
third = fig.add_subplot(3, 1, 3)

first.stem(x_vals,  pdf, label=f"estimate")
# second.stem(x_vals, fixed_pdf/np.linalg.norm(fixed_pdf), label=f"fixed_point")
third.stem(bins_irs[:-1], irs_AED/100, label='True')
plt.legend(loc="upper right")
# ax.grid(True, which='both')
# seaborn.despine(ax=ax, offset=0)  # the important part here
plt.show()