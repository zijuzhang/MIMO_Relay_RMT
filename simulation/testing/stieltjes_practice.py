import numpy as np
import matplotlib.pyplot as plt
from src import circle_laws as cl

size = 500
matrix = np.random.randn(size, size)/np.sqrt(size)
half_circle = (matrix + np.conj(matrix).T)/np.sqrt(2)
evd = np.linalg.eig(half_circle)
eigen_value, bins = np.histogram(evd[0], bins=50)
plt.figure()
plt.title("Half Circle")
plt.bar(bins[0:eigen_value.size], eigen_value/size)

y = 1e-9
values = np.linspace(-2, 2, 100)
# plt.plot(values, true_pdf(values))
plt.plot(values, cl.estimated_pdf(values),'r')
plt.show()
