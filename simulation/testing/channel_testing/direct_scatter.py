from src.circle_laws import *
import numpy as np
import matplotlib.pyplot as plt
from src.lin_alg import *
import seaborn

#   Initialize System Parameters
size = 100
rows = size
cols = size
sigma = 1
rho = 1
phi = 1
#   Plot Theoretical Distribution and find corresponding capacity


def arg_function(s_val, g_k, sigma=1, rho=1):
    x_s = np.sqrt(s_val)
    # return x_s - x_s*g_k/(1-s_val*np.power(g_k, 2))
    return x_s - pow(sigma, 2)*rho*x_s*g_k/(rho-pow(sigma, 2)*s_val*np.power(g_k, 2))


def B_function(x_arg):
    return 1/2*(np.sqrt(1-4/np.power(x_arg, 2))-1)


def final_stieltjes(s_values, g_k, func_params):
    sigma = func_params[0]
    rho = func_params[1]
    x_arg = arg_function(s_values, g_k, sigma=sigma, rho=rho)
    G_BB = (1/2)*np.sqrt(1-(4/np.power(x_arg, 2)))-1/2
    return (1/np.sqrt(s_values))*x_arg*G_BB

def final_stieltjes_deformed(s_values, g_k, func_params):
    sigma = func_params[0]
    rho = func_params[1]
    phi = func_params[2]
    x_arg = arg_function(s_values, g_k, sigma=sigma, rho=rho)

    # Correct
    G_BB = 1/2 + (1-phi)/(2*np.power(x_arg, 2)) +\
           np.sqrt(np.power((1-phi), 2)/(4*np.power(x_arg, 4)) - (1+phi)/(2*np.power(x_arg, 2)) + 1/4)
    # Correct
    # G_BB = np.sqrt(np.power((1-phi), 2)/(4*np.power(x_arg, 4)) - (1+phi)/(2*np.power(x_arg, 2))
    #                + 1/4) - 1/2 - (1-phi)/(2*np.power(x_arg, 2))
    return (1/np.sqrt(s_values))*x_arg*G_BB



x_values, step = np.linspace(1e-2, 10, 100, retstep=True)
y = 1e-6
s_vector = x_values + 1j*y
parameters = [sigma, rho, phi]
theoretic_pdf = estimated_pdf(fixed_point(final_stieltjes_deformed, s_vector, 100, 1e-3, 3, func_param=parameters, unique_half_plane=True))
# theoretic_pdf = estimated_pdf(marcenko_st(s_vector, phi))
theoretic_capacity = aed_capacity(x_values, theoretic_pdf, step, 1/cols, rows)


#   Now check histogram of simulations
bins = 50
irs = []
irs_svd = []
capacities = []
cross_sum = 0
main_sum = 0
los_rank_matrix = reduced_rank(rows, rho)
average = 3
for i in range(average):
    H = sigma*c_rand(rows, cols) @random_phase(rows)@ c_rand(rows, cols)
    # H = c_rand(rows, cols)
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
    # total_irs = GG
    total_channel = H + G
    e_val_total_irs = np.linalg.eigvalsh(total_irs)
    s_val_total_irs = np.linalg.svd(total_channel)[1]
    irs.append(e_val_total_irs)
    irs_svd.append(s_val_total_irs)
    capacities.append(capacity(total_irs, 1/cols))





# check = np.asarray(irs)
# check1 = np.asarray(irs_svd)
irs_AED, bins_irs = np.histogram(np.asarray(irs), bins=bins)
irs_SVD, bins_irs_svd = np.histogram(np.asarray(irs_svd), bins=bins)

#   Compare Estimated to True Capacity
print(f"simulated capacity: {np.average(capacities)}")
print(f"theoretic capacity: {theoretic_capacity}")



#   Plot results
fig, ax = plt.subplots()
plt.title("IRS and Line of Sight: i.i.d, $\mathbb{N}(0,1/N), H = H_1 \Phi H_2$")
ax.bar(bins_irs[1::], irs_AED/rows, label='simulated AED')
# ax.bar(bins_irs_svd[1::], irs_SVD/rows, label='total: ASD')
ax.plot(x_values, normalize(theoretic_pdf), label='normalized theoretical aed', c='r')

plt.legend(loc="upper right")
ax.grid(True, which='both')
seaborn.despine(ax=ax, offset=0)
plt.show()


