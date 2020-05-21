"""
This test is to check the fixed point expression for the concatenated channel from the IRS.
"""

import matplotlib.pyplot as plt
from src.circle_laws import *
from src.fixed_point import *
from src.lin_alg import *
import seaborn

size = 500
rows = 500
cols = 500

def gamma(z, gamma_z):
    return np.power(gamma_z/z, 1/3) - 1

y = 1e-6
x, step = np.linspace(0.01, 10, 1000, retstep=True)
s_vector = x + 1j*y
steiltjes_values = (1/s_vector)*(1+fixed_point(gamma, 1/s_vector, 100, 1e-2))
aed = estimated_pdf(steiltjes_values)
theoretic_capacity = aed_capacity(x, aed, step, 1/cols, rows)


bins = int(size/2)
main_terms = []
no_irs = []
irs = []
numeric_cacacity = []
average = 10
for i in range(average):
    H = c_rand(rows, cols) @ np.diag(np.exp(1j * np.random.uniform(0, 2 * np.pi, rows))) @ c_rand(rows, cols)
    HH = H @ np.conj(H.T)
    total_irs = HH
    e_val_total_irs, e_vec_total_irs = np.linalg.eig(total_irs)
    irs.append(np.real(e_val_total_irs))
    numeric_cacacity.append(capacity(total_irs, 1/cols))



print(f"theoretic_capacity: {theoretic_capacity}")
print(f"numeric_capacity: {np.average(numeric_cacacity)}")

irs_AED, bins_irs = np.histogram(np.asarray(irs), bins=bins)
fig, ax = plt.subplots()
plt.title("IRS Channels Components: i.i.d, $\mathbb{N}(0,1/N), H = H_1 \Phi H_2$")
ax.plot(bins_irs[1::], irs_AED/(rows*average), label='total: with IRS')
ax.plot(x, aed/np.linalg.norm(aed), label='Theoretic AED')

plt.legend(loc="upper right")
ax.grid(True, which='both')
seaborn.despine(ax=ax, offset=0)  # the important part here
plt.show()


