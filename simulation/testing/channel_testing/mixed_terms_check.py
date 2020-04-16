from src.circle_laws import *
import numpy as np
import matplotlib.pyplot as plt
from src.lin_alg import *
import seaborn

size = 500
rows = 500
cols = 500
bins = int(size/2)

main_terms = []
no_irs = []
irs = []
cross = []
cross_irs = []

cross_sum = 0
main_sum = 0

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
    cross_irs_sum = GH + HG
    cross_no_irs_sum = GG + HH
    # cross_no_irs_sum = GF + FG
    main = GG + HH

    e_val_main, e_vec_main = np.linalg.eig(main)
    e_val_total_irs, e_vec_total_irs = np.linalg.eig(total_irs)
    e_val_cross, e_vec_cross = np.linalg.eig(cross_no_irs_sum)
    e_val_cross_irs, e_vec_cross_irs = np.linalg.eig(cross_irs_sum)
    cross_sum += np.sum(e_val_cross_irs)
    main_sum += np.sum(e_val_main)
    cross_irs_vec = eigen_vector_compare_2(e_val_cross, e_vec_cross)
    cross_vec = eigen_vector_compare_2(e_val_cross_irs, e_vec_cross_irs)

    main_terms.append(np.real(e_val_main))
    irs.append(np.real(e_val_total_irs))
    cross.append(np.real(e_val_cross))
    cross_irs.append(np.real(e_val_cross_irs))


print(cross_sum)
print(main_sum)

main_AED, bins_main = np.histogram(np.asarray(main_terms), bins=bins)
irs_AED, bins_irs = np.histogram(np.asarray(irs), bins=bins)
cross_AED, bins_cross_AED = np.histogram(np.asarray(cross), bins=bins)
cross_irs_AED, bins_cross_irs_AED = np.histogram(np.asarray(cross_irs), bins=bins)

fig, ax = plt.subplots()
plt.title("IRS Channels Components: i.i.d, $\mathbb{N}(0,1/N), H = H_1 \Phi H_2$")
values = np.linspace(-2, 2, 200)
# plt.plot(values, semi_cicle_aed(values))
# ax.plot(bins_cross_AED[1::], cross_AED/rows, label='main terms: no IRS')
ax.plot(bins_cross_irs_AED[1::], cross_irs_AED/rows, label='cross terms: with IRS')
ax.plot(bins_main[1::], main_AED/rows, label='main: with IRS')
ax.plot(bins_irs[1::], irs_AED/rows, label='total: with IRS')

plt.legend(loc="upper right")
ax.grid(True, which='both')
seaborn.despine(ax=ax, offset=0)  # the important part here

plt.figure()
plt.plot(np.sort(e_val_cross_irs), label='cross')
plt.plot(np.sort(e_val_main), label='main')
plt.plot(np.sort(e_val_total_irs), label='total')
plt.show()


