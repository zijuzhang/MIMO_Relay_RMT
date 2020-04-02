import numpy as np
import testing.circle_laws as cl
import numpy as np
import matplotlib.pyplot as plt
import seaborn

size = 300
rows = 300
cols = 300
bins = 50
H = np.random.randn(rows, cols) / \
    np.sqrt(rows)@np.diag(np.exp(1j*np.random.uniform(0, 2*np.pi, rows)))@np.random.randn(rows, cols) / np.sqrt(rows)
G = np.random.randn(rows, cols) / np.sqrt(rows)
GG = G @ np.conj(G.T)
GH = G @ np.conj(H.T)
HG = H @ np.conj(G.T)
HH = H @ np.conj(H.T)
total = GH + HG + HH + GG
cross = GH + HG
GG_AED, bins_gg = np.histogram(np.linalg.eig(GG)[0], bins=bins)
HH_AED, bins_hh = np.histogram(np.linalg.eig(HH)[0], bins=bins)
cross_AED, bins_cross = np.histogram(np.linalg.eig(cross)[0], bins=bins)
total_AED, bins_total = np.histogram(np.linalg.eig(total)[0], bins=bins)


average = 15
for i in range(average):
    H = np.random.randn(rows, cols) / \
        np.sqrt(rows) @ np.diag(np.exp(1j * np.random.uniform(0, 2 * np.pi, rows))) @ np.random.randn(rows,
                                                                                                      cols) / np.sqrt(
        rows)
    G = np.random.randn(rows, cols) / np.sqrt(rows)
    GG = G @ np.conj(G.T)
    GH = G @ np.conj(H.T)
    HG = H @ np.conj(G.T)
    HH = H @ np.conj(H.T)
    total = GH + HG + HH + GG
    cross = GH + HG
    GG_AED += np.histogram(np.linalg.eig(GG)[0], bins=bins)[0]
    HH_AED += np.histogram(np.linalg.eig(HH)[0], bins=bins)[0]
    cross_AED += np.histogram(np.linalg.eig(cross)[0], bins=bins)[0]
    total_AED += np.histogram(np.linalg.eig(total)[0], bins=bins)[0]

fig, ax = plt.subplots()
plt.title("AED Comparison: i.i.d, $\mathbb{N}(0,1/N)$, H = H_1 \Phi H_2")

AED = GG_AED/(cols*average)
ax.plot(abs(bins_gg[1::]), AED, label='$GG^H$')

AED = HH_AED/(cols*average)
ax.plot(abs(bins_hh[1::]), AED, label='$HH^H$ (IRS Path)')

AED = cross_AED/(cols*average)
ax.plot(bins_cross[1::], AED, label='$GH^H + HG^H$')

AED = total_AED/(cols*average)
ax.plot(bins_total[1::], AED, label='$(G + H)(G+H)^H$')


values = np.linspace(0, 4, 100)
beta = cols/rows
# ax.plot(values, cl.marcenko(values, beta), label='Marcenko Pastur')
plt.legend(loc="upper right")
ax.grid(True, which='both')
seaborn.despine(ax=ax, offset=0) # the important part here
plt.show()