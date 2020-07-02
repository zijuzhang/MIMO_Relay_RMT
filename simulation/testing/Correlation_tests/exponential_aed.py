from src.lin_alg import *
from src.stieltjes_eqn import *
import matplotlib.pyplot as plt


def gamma_exp_correlation(s, alpha):
    return s/np.sqrt(np.power(s, 2)-2*s*alpha+1)


size = 100
bins = 10
correlations = np.linspace(.1, 0.8, 3)
x_values, step = np.linspace(1e-4, 10, 1000, retstep=True)
s_values = x_values + 1j*1e-6
fig = plt.figure()
aed_plt = fig.add_subplot(2, 1, 1)
num_plt = fig.add_subplot(2, 1, 2)
for rho in correlations:
    alpha = (1+np.power(rho, 2))/(1-np.power(rho, 2))
    steiltjes = (1 / s_values) * (1 + gamma_exp_correlation(s_values, alpha))
    aed = np.abs(estimated_pdf(steiltjes))
    # aed = estimated_pdf(steiltjes)
    aed /= np.linalg.norm(aed)
    aed_plt.plot(x_values, aed, label=f"rho:{rho}")
    r_correlation_matrix = exponential_correlation(size, rho)
    AED, bins_num = np.histogram(np.linalg.eigvalsh(r_correlation_matrix@hermetian(r_correlation_matrix)), bins=bins)
    num_plt.plot(bins_num[:-1], AED/size, label=f"rho:{rho}")

num_plt.title.set_text('Numerical Results')
aed_plt.title.set_text('Asyptotic Results')
# num_plt.set_xlabel("Eigenvalue")
# aed_plt.set_xlabel("Eigenvalue")
num_plt.set_ylabel("Density")
aed_plt.set_ylabel("Density")

num_plt.legend()
aed_plt.legend()
plt.show()



