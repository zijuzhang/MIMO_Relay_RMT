from src.lin_alg import *
import matplotlib.pyplot as plt

num_tx = 50
num_rx = 50
num_reflectors = 50
average = 1000
average = 10
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
transmit_cor = .7
irs_cor = .9
receiver_cor = .1
MSE_zf_mu = []
MSE_matched_mu = []
SINR_matched_mu = []
AR_matched_mu = []
MSE_zf_var = []
MSE_matched_var = []
SINR_matched_var = []
AR_matched_var = []
for theta in phases:
    t_correlation_matrix = exponential_correlation(num_tx, transmit_cor)
    s_correlation_matrix = exponential_correlation(num_reflectors, irs_cor)
    r_correlation_matrix = exponential_correlation(num_rx, receiver_cor)
    phase = np.diag(np.exp(1j * theta * np.ones(num_reflectors)))
    MSE_matched = []
    MSE_zf = []
    SINR_matched = []
    AR_Matched = []
    for j in range(average):
        # transmit_symbols = np.random.randn(num_rx)
        transmit_symbols = c_rand(num_rx, 1, var=1).flatten()
        noise = c_rand(num_rx, 1, var=1).flatten()
        # noise = c_rand(num_rx, 1).flatten()
        G = c_rand(num_reflectors, num_tx, var=1)
        G2 = c_rand(num_rx, num_reflectors, var=1)
        los_path = c_rand(num_rx, num_tx, var=1)
        irs_path = G2 @ s_correlation_matrix @ phase @ G
        channel = r_correlation_matrix @(irs_path + los_path)@ t_correlation_matrix
        transmit_signal_matched = precode_mf(channel) @ transmit_symbols
        received_matched = channel @ transmit_signal_matched + noise
        MSE_matched.append(10*np.log(np.power(np.linalg.norm(received_matched - transmit_symbols), 2)))
        # zf_precoding = precode_zf(channel) @ transmit_symbols
        # received_zf = channel @ zf_precoding + noise
        # MSE_zf.append(10*np.log(np.power(np.linalg.norm(received_zf - transmit_symbols), 2)))
        #   SINR
        signal_power_matched = np.power(np.linalg.norm(channel @ transmit_signal_matched), 2)
        noise_power_matched = np.power(np.linalg.norm(noise), 2)
        SINR_matched.append(10*np.log(signal_power_matched/noise_power_matched))
        #   Signal power over noise power is 1
        AR_Matched.append(np.log(np.linalg.det(np.eye(num_rx) + channel@hermetian(channel))))
    AR_matched_mu.append(np.average(AR_Matched))
    AR_matched_var.append(np.var(AR_Matched))
    SINR_matched_mu.append(np.average(SINR_matched))
    SINR_matched_var.append(np.var(SINR_matched))
    cor_mf_mu.append(np.average(MSE_matched))
    cor_mf_var.append(np.var(MSE_matched))
    cor_zf_mu.append(np.average(MSE_zf))
    cor_zf_var.append(np.var(MSE_zf))

# ave_plt.plot(phases, cor_mf_mu, '-o', label='MSE MF')
# ave_plt.plot(phases, SINR_matched_mu, '-o', label='SINR MF')
ave_plt.plot(phases, AR_matched_mu, '-o', label='Achievable Rate MF')
# ave_plt.plot(phases, cor_zf_mu, '-o', label='Zero-Forcing')
# var_plt.plot(phases, cor_mf_var, '-o', label='MSE MF')
# var_plt.plot(phases, SINR_matched_var, '-o', label='SINR MF')
var_plt.plot(phases, AR_matched_var, '-o', label='Achievable Rate MF')
# var_plt.plot(phases, cor_zf_var, '-o', label='Zero-Forcing')
# ave_plt.set_ylabel("Average MSE  (dB)")
# var_plt.set_ylabel("Average MSE Variance (dB)")
ave_plt.set_ylabel("Achievable Rate")
var_plt.set_ylabel("Achievable Rate Variance ")
# ave_plt.set_yscale('log')
ave_plt.set_xlabel(r"IRS Phase $\mathbf{\Phi} = e^{j\theta} \mathbf{I}, \theta$ = [0,2$\pi$]")
# var_plt.set_yscale('log')
var_plt.set_xlabel(r"IRS Phase $\mathbf{\Phi} = e^{j\theta} \mathbf{I}, \theta$ = [0,2$\pi$]")
ave_plt.legend(loc="upper left")
var_plt.legend(loc="upper left")
ave_plt.set_title(r'Tx $\rho = $' + f"{transmit_cor}," + r'IRS $\rho = $' + f"{irs_cor} and" + r'Rx $\rho = $' + f"{receiver_cor}"
                  + f"Averaged (N={average})")
plt.show()
