# A test for looking at impact of adding correlation through concatenated channels on the channel capacity
# Takeaway:
# If power is allocated equally the total capacity doesn't seem to be affected. However if waterfilling
# is used then the correlated channels seem to have increasing capacity as correlation increases
#
import matplotlib.pyplot as plt
from src.fixed_point import *
from src.lin_alg import *
from src.network_simulation_tools import water_filling
import seaborn


size = 100
rows = 100
cols = 100
bins = int(size/2)
no_irs = []
irs = []
irs_cor = []
rank = 100
phi = rank/rows
average = 5
G_cap = []
cor_cap = []
irs_cap = []
for i in range(average):
    G = c_rand(rows, cols)
    GG = G @ np.conj(G.T)
    e_val_G = np.linalg.eigvalsh(GG)
    no_irs.append(e_val_G)


    H = c_rand(rows, cols) @ np.diag(np.exp(1j * np.random.uniform(0, 2 * np.pi, rows))) @ c_rand(rows, cols)
    HH = H @ np.conj(H.T)
    total_irs = HH
    e_val_total_irs = np.linalg.eigvalsh(total_irs)
    irs.append(np.real(e_val_total_irs))

    # H_cor = c_rand(size, 1)@np.ones((1, size))    # Test keyhole model channel for completely correlated channel
    H_cor = np.eye(size)
    correlations = 10
    for i in range(correlations):
        H_cor = c_rand(size, size)@H_cor
    HH_cor = H_cor @ np.conj(H_cor.T)
    e_val_cor = np.linalg.eigvalsh(HH_cor)
    irs_cor.append(np.real(e_val_cor))

    t1 = np.trace(GG)
    t2 = np.trace(HH)
    t3 = np.trace(HH_cor)

    # G_cap.append(capacity(GG, 1/size))
    # cor_cap.append(capacity(HH_cor, 1/size))
    # irs_cap.append(capacity(HH, 1/size))

    G_cap.append(capacity_water_filled(water_filling(G, 1)))
    cor_cap.append(capacity_water_filled(water_filling(H_cor, 1)))
    irs_cap.append(capacity_water_filled(water_filling(H, 1)))


irs_AED, bins_irs = np.histogram(np.asarray(irs), bins=bins)
AED_cor, bins_cor = np.histogram(np.asarray(irs_cor), bins=bins)
AED_G, bins_G = np.histogram(np.asarray(no_irs), bins=bins)

G_cap_ave = np.average(G_cap)
print(f"los {G_cap_ave}, std: {np.var(G_cap)}")
H_cap_ave = np.average(irs_cap)
print(f"IRS {H_cap_ave}, std: {np.var(irs_cap)}")
H_cor_cap_ave = np.average(cor_cap)
print(f"Correlated: {H_cor_cap_ave}, std: {np.var(cor_cap)}")

fig, ax = plt.subplots()
plt.title("IRS Channels Components: i.i.d, $\mathbb{N}(0,1/N), H = H_1 \Phi H_2$")
ax.plot(bins_irs[:-1], irs_AED/rows, label='total: with IRS')
ax.plot(bins_cor[:-1], AED_cor/rank, label='correlated AED')
ax.plot(bins_G[:-1], AED_G/rank, label='No IRS')


plt.legend(loc="upper right")
ax.grid(True, which='both')
seaborn.despine(ax=ax, offset=0)  # the important part here
plt.show()


