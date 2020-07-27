from src.lin_alg import *
import matplotlib.pyplot as plt

num_tx = 10
num_rx = 10
num_reflectors = 50
average = 100
cor_mf_mu = []
cor_mf_var = []
cor_zf_mu = []
cor_zf_var = []

#   Measure over average
rhos = [.01, .5, .9, .99]
# rhos = [.99]
for cor_rho in rhos:
    MSE_zf_mu = []
    MSE_matched_mu = []
    MSE_zf_var = []
    MSE_matched_var = []
    t_correlation_matrix = exponential_correlation(num_tx, cor_rho)
    s_correlation_matrix = exponential_correlation(num_reflectors, cor_rho)
    r_correlation_matrix = exponential_correlation(num_rx, cor_rho)
    for i in range(average):
        transmit_symbols = np.random.randn(num_rx)
        G = c_rand(num_reflectors, num_tx)
        G2 = c_rand(num_rx, num_reflectors)
        G3 = c_rand(num_rx, num_tx)
        MSE_matched = []
        MSE_zf = []
        noise = c_rand(num_rx, 1, var=1).flatten()
        for j in range(average):
            phase = random_phase(num_reflectors)
            los_path = G3
            irs_path = r_correlation_matrix @ G2 @ s_correlation_matrix @ phase @ G @ t_correlation_matrix
            channel = (irs_path + los_path)
            transmit_signal_matched = precode_mf(channel) @ transmit_symbols
            received_matched = channel @ (transmit_signal_matched + noise)
            MSE_matched.append(np.power(np.linalg.norm(received_matched - transmit_symbols), 2))
            zf_precoding = precode_zf(channel) @ transmit_symbols
            received_zf = channel @(zf_precoding + noise)
            # received_zf = channel @ precode_zf(channel) @ transmit_symbols
            MSE_zf.append(np.power(np.linalg.norm(received_zf - transmit_symbols), 2))
        MSE_zf_mu.append(np.average(MSE_zf))
        MSE_zf_var.append(np.var(MSE_zf))
        MSE_matched_mu.append(np.average(MSE_matched))
        MSE_matched_var.append(np.var(MSE_matched))
    cor_mf_mu.append(np.average(MSE_matched_mu))
    cor_mf_var.append(np.average(MSE_matched_var))
    cor_zf_mu.append(np.average(MSE_zf_mu))
    cor_zf_var.append(np.average(MSE_zf_var))

fig = plt.figure()
ave_plt = fig.add_subplot(2, 1, 1)
var_plt = fig.add_subplot(2, 1, 2)
var_plt.plot(rhos, cor_mf_var, '-o', label=f'Matched Filter')
var_plt.plot(rhos, cor_zf_var, '-o', label=f'Zero-Forcing')
ave_plt.plot(rhos, cor_mf_mu, '-o', label=f'Matched Filter')
ave_plt.plot(rhos, cor_zf_mu, '-o', label=f'Zero-Forcing')
var_plt.set_ylabel("MSE Variance")
ave_plt.set_ylabel("MSE Average")
ave_plt.set_xlabel("Correlation parameter (rho)")
var_plt.set_xlabel("Correlation parameter (rho)")
ave_plt.legend(loc="upper left")
var_plt.legend(loc="upper left")
ave_plt.set_title(f'MSE comparison for N= {average}')


plt.show()
