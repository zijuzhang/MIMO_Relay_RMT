"""
This test is to check the fixed point expression for the concatenated channel from the IRS.
"""

import matplotlib.pyplot as plt
from src.lin_alg import *
import seaborn

size = 100
rows = 100
cols = 100

bins = int(size/2)
main_terms = []
no_irs = []
irs = []
numeric_cacacity = []
average = 10
for i in range(average):
    H = c_bernoulli(rows, cols, .5)
    HH = H @ np.conj(H.T)
    total_irs = HH
    e_val_total_irs, e_vec_total_irs = np.linalg.eig(total_irs)
    irs.append(np.real(e_val_total_irs))

irs_AED, bins_irs = np.histogram(np.asarray(irs), bins=bins)
fig, ax = plt.subplots()
plt.title("AED")
ax.plot(bins_irs[:-1], irs_AED/(rows*average), label='total: with IRS')
# ax.plot(x, normalize_pdf(aed), label='Theoretic AED')   #Only normalizing for visualization but not for capacity calc.
# ax.plot(x, aed, label='Theoretic AED')   #Only normalizing for visualization but not for capacity calc.

plt.legend(loc="upper right")
ax.grid(True, which='both')
seaborn.despine(ax=ax, offset=0)  # the important part here
plt.show()


