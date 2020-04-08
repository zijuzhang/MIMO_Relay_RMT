import numpy as np
import matplotlib.pyplot as plt
from src.circle_laws import *
size = 500
matrix = c_rand(size, size)
half_circle = (matrix + np.conj(matrix).T)/np.sqrt(2)
evd = np.linalg.eig(half_circle)
eigen_value, bins = np.histogram(evd[0], bins=200)
# plt.figure()
# plt.title("Half Circle")
# plt.bar(bins[0:eigen_value.size], eigen_value/size)
# plt.show()

# Analytic comparison to real Steiltjes
H = c_rand(size, size)
X = matrix
P = H@np.conj(H.T)@X
evd = np.linalg.eig(P)
eigen_value, bins = np.histogram(evd[0], bins=50)
plt.figure()
plt.title("Half Circle")
plt.plot(bins[0:eigen_value.size], eigen_value)
plt.show()
#
# plt.figure()
# quarter_circle = matrix@np.conj(matrix.T)
# evd = np.linalg.eig(quarter_circle)
# eigen_value, bins = np.histogram(evd[0], bins=50)
# plt.figure()
# plt.title("Quarter Circle")
# plt.bar(bins[0:eigen_value.size], eigen_value)
# plt.show()