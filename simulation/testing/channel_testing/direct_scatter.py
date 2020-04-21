from src.circle_laws import *
import numpy as np
import matplotlib.pyplot as plt
from src.lin_alg import *
import seaborn

#   Initialize System Parameters
size = 500
rows = size
cols = size

#   Plot Theoretical Distribution and find corresponding capacity

def arg_function(s_val, g_k):
    x_s = np.sqrt(s_val)
    return x_s - x_s*g_k/(1-s_val*np.power(g_k, 2))


def final_stieltjes(s_values, g_k):
    x_arg = arg_function(s_values, g_k)
    return 1/np.sqrt(s_values)*x_arg*((1/2)*np.sqrt(1-4/np.power(x_arg, 2))-1/2)


x_values, step = np.linspace(0.001, 10, 1000, retstep=True)
y = 1e-6
s_vector = x_values + 1j*y
theoretic_pdf = estimated_pdf(fixed_point(final_stieltjes, s_vector, 20, 1e-2, 3))
theoretic_capacity = aed_capacity(x_values, theoretic_pdf, step, 1/cols, rows)


#   Now check histogram of simulations
bins = 50
irs = []
irs_svd = []
capacities = []
cross_sum = 0
main_sum = 0
attenuation_irs = .1
los_rank_ratio = .75
los_rank_matrix = reduced_rank(rows, los_rank_ratio)
average = 1
for i in range(average):
    # H = attenuation_irs*c_rand(rows, cols) @ c_rand(rows, cols)
    H = attenuation_irs*c_rand(rows, cols) @random_phase(rows)@ c_rand(rows, cols)
    G = c_rand(rows, cols)@los_rank_matrix
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
    e_val_total_irs = np.linalg.eigvalsh(total_irs)
    s_val_total_irs = np.linalg.svd(total_channel)[1]
    irs.append(e_val_total_irs)
    irs_svd.append(s_val_total_irs)
    capacities.append(capacity(total_irs))





# check = np.asarray(irs)
# check1 = np.asarray(irs_svd)
irs_AED, bins_irs = np.histogram(np.asarray(irs), bins=bins)
irs_SVD, bins_irs_svd = np.histogram(np.asarray(irs_svd), bins=bins)

#   Compare Estimated to True Capacity
print(np.average(capacities))
# print(theoretic_capacity)



#   Plot results
fig, ax = plt.subplots()
plt.title("IRS and Line of Sight: i.i.d, $\mathbb{N}(0,1/N), H = H_1 \Phi H_2$")
values = np.linspace(-2, 2, 200)
# ax.bar(bins_irs[1::], irs_AED/rows, label='total: AED')
ax.bar(bins_irs_svd[1::], irs_SVD/rows, label='total: ASD')
# ax.plot(x_values, np.abs(theoretic_pdf), label='theoretical distribution')

plt.legend(loc="upper right")
ax.grid(True, which='both')
seaborn.despine(ax=ax, offset=0)
plt.show()


