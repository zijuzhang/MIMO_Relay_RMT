from src.circle_laws import *
import numpy as np
import matplotlib.pyplot as plt
from src.lin_alg import *
import seaborn

#   Initialize System Parameters
size = 100
rows = size
cols = size
sigma = 1   # Relative attenuation of IRS portion
rho = 1     # S/R
phi = 1    # phi=1 is full rank line of sight.  L/T
#   Plot Theoretical Distribution and find corresponding capacity


def arg_function(s_val, g_k, sigma=1, rho=1):
    x_s = np.sqrt(s_val)
    return x_s - (pow(sigma, 2)*rho*x_s*g_k)/(rho-pow(sigma, 2)*s_val*np.power(g_k, 2))


def final_stieltjes_deformed(s_values, g_k, func_params):
    sigma = func_params[0]
    rho = func_params[1]
    phi = func_params[2]
    x_arg = arg_function(s_values, g_k, sigma=sigma, rho=rho)
    # Correct
    G_BB = 1/2 + (1-phi)/(2*np.power(x_arg, 2)) -\
           np.sqrt(np.power((1-phi), 2)/(4*np.power(x_arg, 4)) - (1+phi)/(2*np.power(x_arg, 2)) + 1/4)
    # Correct
    # G_BB = np.sqrt(np.power((1-phi), 2)/(4*np.power(x_arg, 4)) - (1+phi)/(2*np.power(x_arg, 2))
    #                + 1/4) - 1/2 - (1-phi)/(2*np.power(x_arg, 2))
    return (1/np.sqrt(s_values))*x_arg*G_BB





x_values, step = np.linspace(1e-2, 10, 100, retstep=True)
y = 1e-9
s_vector = x_values + 1j*y

numeric_capacity_list = []
theoretic_capacity_list = []
param_list = [0, .25 , .5 , 1]


for val in param_list:
    phi = val
    # sigma = val
    parameters = [sigma, rho, phi]
    theoretic_pdf = estimated_pdf(fixed_point(final_stieltjes_deformed, s_vector, 100, 1e-3, 3,
                                              func_param=parameters, unique_half_plane=True))
    # theoretic_pdf = estimated_pdf(marcenko_st(s_vector, phi))
    theoretic_capacity = aed_capacity(x_values, theoretic_pdf, step, 1/cols, rows)


    #   Now check histogram of simulations
    bins = 50
    irs = []
    irs_svd = []
    capacities = []
    cross_sum = 0
    main_sum = 0
    los_rank_matrix = reduced_rank(rows, phi)
    average = 1
    for i in range(average):
        # H = sigma*c_rand(rows, cols) @random_phase(rows)@ c_rand(rows, cols)
        H = sigma*c_rand(rows, cols) @ c_rand(rows, cols)
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
        total_irs = GH + HG + GG + HH
        # total_irs = GG
        total_channel = H + G
        e_val_total_irs = np.linalg.eigvalsh(total_irs)
        s_val_total_irs = np.linalg.svd(total_channel)[1]
        irs.append(e_val_total_irs)
        irs_svd.append(s_val_total_irs)
        capacities.append(capacity(total_irs, 1/cols))

#   Compare Estimated to True Capacity
    numeric_capacity_list.append(np.average(capacities))
    theoretic_capacity_list.append(theoretic_capacity)



#   Plot results
fig, ax = plt.subplots()
plt.title("Capacity Vs. Line of Sight Rank")
# plt.title("Capacity Vs. IRS Path Attenuation")
ax.plot(param_list, theoretic_capacity_list, label='theoretical capacity', c='g')
ax.plot(param_list, numeric_capacity_list, label='numerical capacity (N=100)', c='r')
ax.set_ylabel('Capacity (Bits)')
# ax.set_xlabel('Relative Attenuation Coefficient')
ax.set_xlabel('Line of Sight Channel Rank')
plt.legend(loc="upper left")
ax.grid(True, which='both')
seaborn.despine(ax=ax, offset=0)
plt.show()


