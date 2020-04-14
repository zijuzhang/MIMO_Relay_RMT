from src.circle_laws import *
import numpy as np
import matplotlib.pyplot as plt
from src.lin_alg import *
import seaborn

size = 100
rows = 100
cols = 100
bins = int(size/2)


no_irs = []
irs = []
cross = []
cross_irs = []
average = 10
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
    cross_irs_sum = GH + HG
    cross_no_irs_sum = GG + HH
    # cross_irs_sum = FF + GG
    # cross_no_irs_sum = GF + FG
    main = GG + HH

    e_val_total, e_vec_total = np.linalg.eig(total)
    e_val_total_irs, e_vec_total_irs = np.linalg.eig(total_irs)
    e_val_cross, e_vec_cross = np.linalg.eig(cross_no_irs_sum)
    e_val_cross_irs, e_vec_cross_irs = np.linalg.eig(cross_irs_sum)

    cross_irs_vec = eigen_vector_compare_2(e_val_cross, e_vec_cross)
    cross_vec = eigen_vector_compare_2(e_val_cross_irs, e_vec_cross_irs)

    no_irs.append(np.real(e_val_total))
    irs.append(np.real(e_val_total_irs))
    cross.append(np.real(e_val_cross))
    cross_irs.append(np.real(e_val_cross_irs))

no_irs_AED, bins_no_irs = np.histogram(np.asarray(no_irs), bins=bins)
irs_AED, bins_irs = np.histogram(np.asarray(irs), bins=bins)
cross_AED, bins_cross_AED = np.histogram(np.asarray(cross), bins=bins)
cross_irs_AED, bins_cross_irs_AED = np.histogram(np.asarray(cross_irs), bins=bins)

fig, ax = plt.subplots()
plt.title("IRS Channels Components: i.i.d, $\mathbb{N}(0,1/N), H = H_1 \Phi H_2$")
values = np.linspace(-2, 2, 200)
plt.plot(values, semi_cicle_aed(values))
ax.plot(bins_cross_AED[1::], cross_AED/rows, label='main terms: no IRS')
ax.plot(bins_cross_irs_AED[1::], cross_irs_AED/rows, label='cross terms: with IRS')
# ax.plot(bins_no_irs[1::], no_irs_AED/rows, label='total: no IRS')
ax.plot(bins_irs[1::], irs_AED/rows, label='total: with IRS')

plt.legend(loc="upper right")
ax.grid(True, which='both')
seaborn.despine(ax=ax, offset=0)  # the important part here

# plt.figure()
# plt.plot(cross_vec)
# plt.plot(cross_irs_vec)

plt.show()