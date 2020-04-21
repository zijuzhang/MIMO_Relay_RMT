import numpy as np
import matplotlib.pyplot as plt
from src.circle_laws import *
from src.lin_alg import *
rows = 500
cols = 500
bins = 50
alpha = cols/rows
beta = rows/cols
los_rank = 1
los_rank_matrix = reduced_rank(rows, los_rank)
H_los = c_rand(rows, cols)@los_rank_matrix
H_scatter = .25*c_rand(rows, cols)@c_rand(rows, cols)
total = H_los@np.conj(H_los).T
# svd = np.linalg.svd(H_scatter+H_los)
# eigen_value, bins = np.histogram(svd[1], bins=bins)
evd = np.linalg.eigvalsh(total)
eigen_value, bins = np.histogram(evd, bins=bins)
values, step = np.linspace(1e-2, 4, 100, retstep=True)
distribution = deformed_qc_eigen(values, alpha)
st_distribution = estimated_pdf(st_quarter_circle_eigen(values+1j*1e-6))
print(f" simulation capacity: {capacity(total,1/cols)}")
print(f"theoretic capacity: {aed_capacity(values, st_distribution, step, 1/cols, rows)}")
plt.figure()
plt.title("deformed quarter circle")
plt.plot(values, st_distribution)
plt.plot(values, distribution)
plt.plot(bins[1::], eigen_value/rows)
plt.show()

