from src.circle_laws import *
import numpy as np
import matplotlib.pyplot as plt
from src.lin_alg import *
import seaborn

#   First Plot Theoretical Distribution


def arg_function(s_val, g_k):
    x_s = np.sqrt(s_val)
    return x_s - x_s*g_k/(1-s_val*np.power(g_k, 2))


def final_stieltjes(s_values, g_k):
    x_arg = arg_function(s_values, g_k)
    return 1/np.sqrt(s_values)*x_arg*((1/2)*np.sqrt(1-4/np.power(x_arg, 2))-1/2)


x_values = np.linspace(0.001, 10, 1000)
y = 1e-6
s_vector = x_values + 1j*y
# theoretic_pdf = estimated_pdf(fixed_point(final_stieltjes, s_vector, 20, 1e-2, 3))



#   Now check histogram of simulations

size = 500
rows = size
cols = size
bins = int(size/2)
irs = []
irs_svd = []
cross_sum = 0
main_sum = 0
attenuation_irs = .21
average = 2
for i in range(average):
    H = c_rand(rows, cols) @ c_rand(rows, cols)
    # H = c_rand(rows, cols) @ np.diag(np.exp(1j * np.random.uniform(0, 2 * np.pi, rows))) @ c_rand(rows, cols)
    G = c_rand(rows, cols)
    # check = np.var(H)
    # check1 = np.var(G)
    F = c_rand(rows, cols)
    GG = G @ np.conj(G.T)
    HH = H @ np.conj(H.T)
    FF = F @ np.conj(F.T)
    GH = G @ np.conj(H.T)
    HG = H @ np.conj(G.T)
    GF = G @ np.conj(F.T)
    FG = F @ np.conj(G.T)
    total_irs = GH + HG + HH + GG
    total_channel = H + G
    e_val_total_irs, e_vec_total_irs = np.linalg.eig(total_irs)
    s_val_total_irs = np.linalg.svd(total_channel)[1]

    irs.append(np.real(e_val_total_irs))
    irs_svd.append(s_val_total_irs)

irs_AED, bins_irs = np.histogram(np.asarray(irs), bins=bins)
irs_SVD, bins_irs_svd = np.histogram(np.asarray(irs_svd), bins=bins)



#   Plot results
fig, ax = plt.subplots()
plt.title("IRS and Line of Sight AED: i.i.d, $\mathbb{N}(0,1/N), H = H_1 \Phi H_2$")
values = np.linspace(-2, 2, 200)
# ax.plot(bins_irs[1::], irs_AED/rows, label='total: AED')
# ax.plot(bins_irs_svd[1::], irs_SVD/rows, label='total: ASD')

# ax.plot(x_values, np.abs(theoretic_pdf), label='theoretical distribution')

plt.legend(loc="upper right")
ax.grid(True, which='both')
seaborn.despine(ax=ax, offset=0)

plt.figure()
plt.plot(irs_SVD)

plt.show()

