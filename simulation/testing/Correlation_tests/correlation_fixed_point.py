# A test for looking at impact of adding correlation through concatenated channels on the channel capacity
# Takeaway:
# If power is allocated equally the ave capacity doesn't seem to be affected however variance increase
# with increasing correlation. However if water-filling
# is used then the correlated channels seem to have increasing capacity as correlation increases

import matplotlib.pyplot as plt
from src.fixed_point import *
from src.lin_alg import *
from src.circle_laws import estimated_pdf
import seaborn
from scipy.linalg import toeplitz

rows = 100
cols = 100
H_cor = c_rand(rows, cols)
block = toeplitz(np.array((1, .5)))
correlation_matrix = block_matrix(block, H_cor.shape[0], normalize=True)
e_val_correlation = np.linalg.eigvalsh(correlation_matrix)
H_cor = correlation_matrix@H_cor
HH = H_cor@hermetian(H_cor)
e_val = np.linalg.eigvalsh(HH)
AED, bins = np.histogram(np.asarray(e_val), bins=50)
x_values = np.linspace(1e-6, 5, 100)
s_values = x_values + 1j*1e-6

def gamma_c_fixed(s, gamma_s):
    """
    Using an out of scope variable here because I don't want to pass it though another fixed point equation.
    :param s:
    :param gamma_s:
    :return:
    """
    num = e_val_correlation.size
    total = 0
    for eigen_val in e_val_correlation:
        total += (1/num)*(1/(eigen_val-1/gamma_s))
    total /= (-(1+s))
    if np.isnan(total) or np.isinf(total) or total <= 1e-6:
        print("nan")
    return total

def S_ch(gamma_s):
    check = fixed_point(gamma_c_fixed, gamma_s, 100, tolerance=1e-3)*(1/gamma_s)
    #TODO current problem is that gamma_s is zero
    if np.isnan(check) or np.isinf(check):
        print("catch")
    return check

def gammafixed(s, gamma_s):
    # check = S_ch(gamma_s)
    if np.isinf(S_ch(gamma_s)) or np.isnan(S_ch(gamma_s)):
        print("inf.nan")
    return s*S_ch(gamma_s)/(1+gamma_s)


def gammavals(s_values):
    ret = fixed_point(gammafixed, s_values, 100)
    return ret

stieltjes_values = -1/(s_values)*(1+gammavals(s_values))
pdf = estimated_pdf(stieltjes_values)

fig, ax = plt.subplots()
plt.title("IRS Channels Components: i.i.d, $\mathbb{N}(0,1/N), H = H_1 \Phi H_2$")
ax.plot(x_values, pdf, label='asymptotic')
# ax.plot(bins[:-1], AED/rows, label='total: with IRS')
plt.legend(loc="upper right")
ax.grid(True, which='both')
seaborn.despine(ax=ax, offset=0)
# plt.show()


