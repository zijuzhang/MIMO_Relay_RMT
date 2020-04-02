import numpy as np
import testing.circle_laws as cl
import numpy as np
import matplotlib.pyplot as plt
import seaborn


rows = 500
cols = 400
bins = 50
H = np.random.randn(rows, cols) / np.sqrt(rows)
G = np.random.randn(rows, cols) / np.sqrt(rows)
GG = G @ np.conj(G.T)
GH = G @ np.conj(H.T)
HG = H @ np.conj(G.T)
HH = H @ np.conj(H.T)
total = GH + HG + HH + GG
cross = GH + HG
GG_AED, bins = np.histogram(np.linalg.eig(GG)[0], bins=bins)
HH_AED, bins = np.histogram(np.linalg.eig(HH)[0], bins=bins)
cross_AED, bins = np.histogram(np.linalg.eig(cross)[0], bins=bins)
total_AED, bins = np.histogram(np.linalg.eig(total)[0], bins=bins)


average = 1
for i in range(average):
    H = np.random.randn(rows, cols) / np.sqrt(rows)
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
plt.title("AED Comparison: $(G + H)(G+H)^H$ ")

AED = GG_AED/(cols*average)
ax.plot(abs(bins[1::]), AED, label='$GG^H$')

AED = HH_AED/(cols*average)
ax.plot(abs(bins[1::]), AED, label='$HH^H$')

AED = cross_AED/(cols*average)
ax.plot(bins[1::], AED, label='$GH^H + HG^H$')

AED = total_AED/(cols*average)
ax.plot(bins[1::], AED, label='$(G + H)(G+H)^H$')


values = np.linspace(0, 4, 100)
beta = cols/rows
# ax.plot(values, cl.marcenko(values, beta), label='Marcenko Pastur')
plt.legend(loc="upper right")
ax.grid(True, which='both')
seaborn.despine(ax=ax, offset=0) # the important part here
plt.show()