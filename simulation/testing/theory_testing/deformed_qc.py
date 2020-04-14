import numpy as np
import matplotlib.pyplot as plt
from src.circle_laws import *
from src.lin_alg import *
rows = 500
cols = 500
bins = int(rows/2)
alpha = cols/rows
beta = rows/cols
H = c_rand(rows, cols)
total = H@np.conj(H).T
evd = np.linalg.eig(total)
eigen_value, bins = np.histogram(evd[0], bins=bins)
values = np.linspace(0, 4, 100)
plt.figure()
plt.title("deformed quarter circle")
plt.plot(values, deformed_qc_eigen(values, alpha))
plt.plot(values, marcenko(values, alpha))
plt.plot(bins[1::], eigen_value/rows)
plt.show()

