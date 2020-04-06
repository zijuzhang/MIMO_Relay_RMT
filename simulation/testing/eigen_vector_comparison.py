import numpy as np
from testing.circle_laws import *
import testing.circle_laws as cl
import numpy as np
import matplotlib.pyplot as plt
import seaborn

size = 500
rows = 500
cols = 500
bins = 300

fig, ax = plt.subplots()
plt.title("Eigen Vector Comparison: i.i.d, $\mathbb{N}(0,1/N)$, H = H_1 \Phi H_2")

average = 1
for i in range(average):
    H = c_rand(rows, cols) @ np.diag(np.exp(1j * np.random.uniform(0, 2 * np.pi, rows))) @ c_rand(rows, cols)
    G = c_rand(rows, cols)
    F = c_rand(rows, cols)
    GG = G @ np.conj(G.T)
    HH = H @ np.conj(H.T)
    FF = F @ np.conj(F.T)
    GH = G @ np.conj(H.T)
    HG = H @ np.conj(G.T)
    GF = G @ np.conj(F.T)
    FG = F @ np.conj(G.T)
    total_irs = GH + HG + HH + GG
    total = GF + FG + FF + GG
    check = GG + FF
    cross = GH + HG
    main = GG + HH
    e_val_total, e_vec_total = np.linalg.eig(total)
    e_val_total_irs, e_vec_total_irs = np.linalg.eig(total_irs)
    total_max = np.max(e_val_total)
    irs_max = np.max(e_val_total_irs)
    ax.plot(e_val_total, label=f'no_irs_phase', color='r')
    ax.plot(e_val_total_irs, label=f'irs path', color='g')

plt.legend(loc="upper right")
ax.grid(True, which='both')
seaborn.despine(ax=ax, offset=0) # the important part here
plt.show()