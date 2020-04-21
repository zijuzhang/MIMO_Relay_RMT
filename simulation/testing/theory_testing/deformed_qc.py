import numpy as np
import matplotlib.pyplot as plt
from src.circle_laws import *
from src.lin_alg import *
rows = 500
cols = 500
bins = 50
alpha = cols/rows
beta = rows/cols
los_rank = .5
los_rank_vector = 1j*np.ones(rows)
los_rank_vector[int(los_rank_vector.size*los_rank):los_rank_vector.size] = 0
los_rank_matrix = np.diag(los_rank_vector)
H_los = c_rand(rows, cols)@los_rank_matrix
H_scatter = .25*c_rand(rows, cols)@c_rand(rows, cols)
total = H_los@np.conj(H_los).T
svd = np.linalg.svd(H_scatter+H_los)
# svd = np.linalg.svd(H_scatter)
eigen_value, bins = np.histogram(svd[1], bins=bins)
values = np.linspace(0, 10, 100)
plt.figure()
plt.title("deformed quarter circle")
# plt.plot(values, deformed_qc_eigen(values, alpha))
# plt.plot(values, marcenko(values, alpha))
plt.plot(bins[1::], eigen_value/rows)
plt.show()

