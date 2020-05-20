import matplotlib.pyplot as plt
from src.circle_laws import *
from src.fixed_point import *
from src.lin_alg import *
import seaborn


def gamma(z, gamma_z):
    return np.power(gamma_z/z, 1/3) - 1
y = 1e-6
x = np.linspace(0.01, 30, 1000)
s_vector = x + 1j*y
steiltjes_values = (-1/s_vector)*(fixed_point(gamma, 1/s_vector, 20, 1e-2)+1)
aed = estimated_pdf(steiltjes_values)


size = 500
rows = 500
cols = 500
bins = int(size/2)
main_terms = []
no_irs = []
irs = []

average = 1
for i in range(average):
    H = c_rand(rows, cols) @ np.diag(np.exp(1j * np.random.uniform(0, 2 * np.pi, rows))) @ c_rand(rows, cols)
    HH = H @ np.conj(H.T)
    total_irs = HH
    e_val_total_irs, e_vec_total_irs = np.linalg.eig(total_irs)
    irs.append(np.real(e_val_total_irs))

irs_AED, bins_irs = np.histogram(np.asarray(irs), bins=bins)


fig, ax = plt.subplots()
plt.title("IRS Channels Components: i.i.d, $\mathbb{N}(0,1/N), H = H_1 \Phi H_2$")
ax.plot(bins_irs[1::], irs_AED/rows, label='total: with IRS')
ax.plot(x, aed, label='Theoretic AED')

plt.legend(loc="upper right")
ax.grid(True, which='both')
seaborn.despine(ax=ax, offset=0)  # the important part here
plt.show()


