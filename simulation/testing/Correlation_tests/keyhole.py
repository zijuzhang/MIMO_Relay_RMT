import matplotlib.pyplot as plt
from src.fixed_point import *
from src.lin_alg import *
import seaborn


size = 100
rows = 100
cols = 100
bins = int(size/2)
no_irs = []
irs = []
irs_keyhole = []
rank = 100
phi = rank/rows
average = 5
for i in range(average):
    G = c_rand(rows, cols)
    GG = G @ np.conj(G.T)
    e_val_G = np.linalg.eigvalsh(GG)
    no_irs.append(e_val_G)


    H = c_rand(rows, cols, ) @ np.diag(np.exp(1j * np.random.uniform(0, 2 * np.pi, rows)))\
        @reduced_rank(rows, phi) @ c_rand(rows, cols, 1/(2*np.power(rows, 1)))
    HH = H @ np.conj(H.T)
    total_irs = HH
    e_val_total_irs = np.linalg.eigvalsh(total_irs)
    irs.append(np.real(e_val_total_irs))

    H_keyhole = c_rand(rows, 1)@c_rand(1, cols)*0
    for i in range(rank):
        H_keyhole += c_rand(rows, 1, 1/(2*np.power(rows, 2))) @ c_rand(1, cols, 1/(2*np.power(rows, 2)))
    HH_keyhole = H_keyhole @ np.conj(H_keyhole.T)
    e_val_keyhole = np.linalg.eigvalsh(HH_keyhole)
    irs_keyhole.append(np.real(e_val_keyhole))

    t1 = np.trace(GG)
    t2 = np.trace(HH)
    t3 = np.trace(HH_keyhole)
    print("check")


irs_AED, bins_irs = np.histogram(np.asarray(irs), bins=bins)
AED_keyhole, bins_keyhole = np.histogram(np.asarray(irs_keyhole), bins=bins)
AED_G, bins_G = np.histogram(np.asarray(no_irs), bins=bins)


fig, ax = plt.subplots()
plt.title("IRS Channels Components: i.i.d, $\mathbb{N}(0,1/N), H = H_1 \Phi H_2$")
ax.plot(bins_irs[1::], irs_AED/rows, label='total: with IRS')
# ax.plot(bins_keyhole[1::], AED_keyhole/rank, label='keyhole AED')
ax.plot(bins_G[1::], AED_G/rank, label='No IRS')


plt.legend(loc="upper right")
ax.grid(True, which='both')
seaborn.despine(ax=ax, offset=0)  # the important part here
plt.show()


