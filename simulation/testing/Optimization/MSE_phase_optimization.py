from src.lin_alg import *
import matplotlib.pyplot as plt

num_tx = 16
num_rx = 10
num_reflectors = 128
average = 1000
cor_mf_mu = []
cor_mf_var = []
cor_zf_mu = []
cor_zf_var = []
#   Measure over average
rhos = [.1, .5, .7, .9, .99]
phases = np.linspace(0, 2 * np.pi, 10)
fig = plt.figure()
ave_plt = fig.add_subplot(2, 1, 1)
var_plt = fig.add_subplot(2, 1, 2)
for cor_rho in rhos:
    t_correlation_matrix = exponential_correlation(num_tx, cor_rho)
    s_correlation_matrix = exponential_correlation(num_reflectors, cor_rho)
    r_correlation_matrix = exponential_correlation(num_rx, cor_rho)
    MSE_zf_mu = []
    MSE_matched_mu = []
    MSE_zf_var = []
    MSE_matched_var = []
    for j in range(average):
        MSE_matched = []
        MSE_zf = []
        for theta in phases:
            phase = np.diag(np.exp(1j * theta * np.ones(num_reflectors)))
            # transmit_symbols = np.random.randn(num_rx)
            transmit_symbols = c_rand(num_rx, 1).flatten()
            # noise = c_rand(num_rx, 1, var=1).flatten()
            noise = c_rand(num_rx, 1).flatten()
            G = c_rand(num_reflectors, num_tx)
            G2 = c_rand(num_rx, num_reflectors)
            los_path = c_rand(num_rx, num_tx)
            irs_path = G2 @ s_correlation_matrix @ phase @ G
            channel = r_correlation_matrix @(irs_path + los_path)@ t_correlation_matrix
            transmit_signal_matched = precode_mf(channel) @ transmit_symbols
            received_matched = channel @ transmit_signal_matched + noise
            MSE_matched.append(10*np.log(np.power(np.linalg.norm(received_matched - transmit_symbols), 2)))
            zf_precoding = precode_zf(channel) @ transmit_symbols
            received_zf = channel @zf_precoding + noise
            MSE_zf.append(10*np.log(np.power(np.linalg.norm(received_zf - transmit_symbols), 2)))
        MSE_zf_mu.append(np.average(MSE_zf))
        MSE_zf_var.append(np.var(MSE_zf))
        MSE_matched_mu.append(np.average(MSE_matched))
        MSE_matched_var.append(np.var(MSE_matched))
    cor_mf_mu.append(np.average(MSE_matched_mu))
    cor_mf_var.append(np.average(MSE_matched_var))
    cor_zf_mu.append(np.average(MSE_zf_mu))
    cor_zf_var.append(np.average(MSE_matched_mu))

ave_plt.plot(rhos, cor_mf_mu, '-o', label='Matched Filter')
ave_plt.plot(rhos, cor_zf_mu, '-o', label='Zero-Forcing')
var_plt.plot(rhos, cor_mf_var, '-o', label='Matched Filter')
var_plt.plot(rhos, cor_zf_var, '-o', label='Zero-Forcing')
ave_plt.set_ylabel("Average MSE  (dB)")
var_plt.set_ylabel("Average MSE Variance (dB)")
ave_plt.set_yscale('log')
ave_plt.set_xlabel("Correlation Parameter (rho)")
var_plt.set_yscale('log')
var_plt.set_xlabel("Correlation Parameter (rho)")
ave_plt.legend(loc="upper left")
var_plt.legend(loc="upper left")
ave_plt.set_title(f'MSE comparison for N = {average}')


plt.show()
