import numpy as np
from src.lin_alg import *
from src.stieltjes_eqn import estimated_pdf
import matplotlib.pyplot as plt
import seaborn


def deterministic_stieltjes(s_vals, e_vals):
    num = e_vals.size
    total = 1j*np.zeros(s_vals.shape)
    for e_val in e_vals:
        total += 1/((e_val-s_vals)*num)
    return total


#   First Generate a deterministic matric
matrix = c_rand(100, 100)
# matrix = np.random.randint(0, 10, (100, 100))
matrix = matrix@hermetian(matrix)
e_vals, e_vec = np.linalg.eigh(matrix)

# x_vals = np.linspace(0, 5, 100)
x_vals  = e_vals
s_vals = x_vals + 1j*1e-6
pdf = estimated_pdf(deterministic_stieltjes(s_vals, e_vals))
irs_AED, bins_irs = np.histogram(e_vals, bins=50)

fig, ax = plt.subplots()
ax.stem(x_vals, pdf/np.linalg.norm(pdf), label=f"estimate")
ax.plot(bins_irs[:-1], irs_AED/100, label='True')
plt.legend(loc="upper right")
ax.grid(True, which='both')
seaborn.despine(ax=ax, offset=0)  # the important part here
plt.show()