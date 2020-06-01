# A test for looking at impact of adding correlation through concatenated channels on the channel capacity
# Takeaway:
# If power is allocated equally the ave capacity doesn't seem to be affected however variance increase
# with increasing correlation. However if water-filling
# is used then the correlated channels seem to have increasing capacity as correlation increases

import matplotlib.pyplot as plt
from src.lin_alg import *
from src.stieltjes_eqn import *
import seaborn
from scipy.linalg import toeplitz
from src.circle_laws import *
from scipy.optimize import *
from scipy.optimize import fsolve


rows = 100
cols = 100
x_values, step = np.linspace(.01, 10, 100, retstep=True)
s_values = x_values + 1j*1e-6

def R_Transform(s, past):
    return 1/(1/(1+past) - s)

stieltjes_values = steiltjes_from_r_transform(s_values, R_Transform)
pdf = estimated_pdf(stieltjes_values)
print(f"asymptotic capacity: {aed_capacity(x_values, pdf, 1/rows, rows, step=step)}")

fig, ax = plt.subplots()
plt.title("IRS Channels Components: i.i.d, $\mathbb{N}(0,1/N), H = H_1 \Phi H_2$")
ax.plot(x_values, pdf, label='asymptotic')
plt.legend(loc="upper right")
ax.grid(True, which='both')
seaborn.despine(ax=ax, offset=0)
plt.show()


