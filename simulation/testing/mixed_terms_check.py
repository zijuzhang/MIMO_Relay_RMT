from src.circle_laws import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn

size = 500
rows = 500
cols = 500
bins = 300

fig, ax = plt.subplots()
plt.title("Eigen Vector Comparison: i.i.d, $\mathbb{N}(0,1/N)$, H = H_1 \Phi H_2")

no_irs = []
irs = []
main_none_list = []
main_irs_list = []
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
    cross_no_irs_sum = GF + FG
    main = GG + FF
    main_irs = GG + HH
    e_val_total, e_vec_total = np.linalg.eig(total)
    e_val_total_irs, e_vec_total_irs = np.linalg.eig(total_irs)
    e_val_main, e_vec_total = np.linalg.eig(main)
    e_val_main_irs, e_vec_total = np.linalg.eig(main_irs)

    no_irs.append(np.real(e_val_total))
    irs.append(np.real(e_val_total_irs))
    main_none_list.append(np.real(e_val_main))
    main_irs_list.append(np.real(e_val_main_irs))

no_irs_AED, bins_no_irs = np.histogram(np.asarray(no_irs), bins=bins)
irs_AED, bins_irs = np.histogram(np.asarray(irs), bins=bins)
main_AED, bins_cross_AED = np.histogram(np.asarray(main), bins=bins)
main_irs_AED, bins_cross_irs_AED = np.histogram(np.asarray(main_irs), bins=bins)

ax.plot(bins_cross_AED[1::], main_AED/rows, label='main no IRS')
ax.plot(bins_cross_irs_AED[1::], main_irs_AED/rows, label='main with IRS')
ax.plot(bins_no_irs[1::], no_irs_AED/rows, label='no IRS')
ax.plot(bins_irs[1::], irs_AED/rows, label='with IRS')


values = np.linspace(0, 4, 100)
beta = cols/rows
# ax.plot(values, cl.marcenko(values, beta), label='Marcenko Pastur')
plt.legend(loc="upper right")
ax.grid(True, which='both')
seaborn.despine(ax=ax, offset=0) # the important part here
plt.show()