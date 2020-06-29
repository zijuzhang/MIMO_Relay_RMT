# About:
#   Test comparing the AED of the different terms from a scattering/LOS channel.
# Takeaways
#   While the mixed terms matrix sum has real eigenvalues, in the zero-mean case they do not impact the final AED.



import matplotlib.pyplot as plt
from src.lin_alg import *
import seaborn

size = 100
rows = 100
cols = 100
bins = int(size/2)


irs = []
cross_e = []
main_e = []
irs_path = []
los_path = []
cross_sum = 0
main_sum = 0

average = 50
for i in range(average):
    # H = c_rand(rows, cols) @ np.diag(np.exp(1j * np.random.uniform(0, 2 * np.pi, rows))) @ c_rand(rows, cols)
    H = c_rand(rows, cols) @ np.diag(np.exp(1j * np.random.uniform(0, 2 * np.pi, rows))) @ c_rand(rows, cols)
    # G = c_rand(rows, cols, mean=.1)
    G = np.ones((rows, cols))*.05   
    # G = np.zeros((rows, cols))*-.01
    # G = c_rand(rows, cols)
    GG = G @ np.conj(G.T)
    HH = H @ np.conj(H.T)
    GH = G @ np.conj(H.T)
    HG = H @ np.conj(G.T)
    cross = GH + HG
    main = HH + GG
    total_irs = cross + main
    trc = np.trace(total_irs)
    check = np.linalg.eigh(total_irs)
    check1 = check[1][:, -1]
    e_val_total_irs = np.linalg.eigvalsh(total_irs)
    e_val_main = np.linalg.eigvalsh(main)
    e_val_los = np.linalg.eigvalsh(GG)
    e_val_irs = np.linalg.eigvalsh(HH)
    e_val_cross = np.linalg.eigvalsh(cross)
    cross_sum += np.sum(e_val_cross)
    main_sum += np.sum(e_val_total_irs)
    irs.append(e_val_total_irs)
    cross_e.append(e_val_cross)
    main_e.append(e_val_main)
    los_path.append(e_val_los)
    irs_path.append(e_val_irs)


irs_AED, bins_irs = np.histogram(np.asarray(irs), bins=bins)
irs_AED_cross, bins_irs_cross = np.histogram(np.asarray(cross_e), bins=bins)
irs_AED_main, bins_irs_main = np.histogram(np.asarray(main_e), bins=bins)
irs_path, bins_irs_path = np.histogram(np.asarray(irs_path), bins=bins)
los_AED_main, bins_los = np.histogram(np.asarray(los_path), bins=bins)


print(f"cross total: {cross_sum}")
print(f"main total: {main_sum}")

fig, ax = plt.subplots()
plt.title("IRS Channels Components: All channels are i.i.d, $\mathbb{CN}(0,1/N)$")
values = np.linspace(-2, 2, 200)
ax.plot(bins_irs[:-1], irs_AED/rows, label='total: with IRS')
ax.plot(bins_irs_cross[:-1], irs_AED_cross/rows, label='cross terms: with IRS')
ax.plot(bins_irs_main[:-1], irs_AED_main/rows, label='non-cross terms: with IRS')
ax.plot(bins_irs_path[:-1], irs_path/rows, label='irs path')
ax.plot(bins_los[:-1], los_AED_main/rows, label='los')


plt.legend(loc="upper right")
ax.grid(True, which='both')
seaborn.despine(ax=ax, offset=0)  # the important part here

# plt.figure()
# plt.plot(np.sort(e_val_total_irs), label='total')
plt.show()


