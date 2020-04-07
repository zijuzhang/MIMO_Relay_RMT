import src.circle_laws as cl
import numpy as np
import matplotlib.pyplot as plt
rows = 500
cols = 500
size = 500
num_irs = 1
matrix = np.eye(size)
matrix_irs = np.eye(size)
for i in range(num_irs):
    channel = np.random.randn(rows, cols)

    irs_phase = np.random.uniform(0, 2*np.pi,size)
    irs_phase = np.diag(np.exp(1j*irs_phase))
    channel_irs = irs_phase@np.random.randn(rows, cols)
    matrix = channel@matrix/np.sqrt(size)
    matrix_irs = channel_irs@matrix/np.sqrt(size)


covariance = matrix@np.conj(matrix.T)
svd = np.linalg.svd(covariance)
eigen_value, bins = np.histogram(svd[1], bins=50)
plt.figure()
plt.subplot(1, 3, 1)
plt.title("concatenated")
plt.bar(abs(bins[0:eigen_value.size]), eigen_value)


covariance = matrix_irs@np.conj(matrix_irs.T)
svd = np.linalg.svd(covariance)
eigen_value, bins = np.histogram(svd[1], bins=50)
plt.subplot(1, 3, 2)
plt.title("irs")
plt.bar(bins[0:eigen_value.size], eigen_value)

plt.subplot(1, 3, 3)
values = np.linspace(0, 4, 100)
plt.plot(values, cl.estimated_pdf(cl.st_quarter_circle(values+1j*1e-9)), 'r')
beta = rows/cols
plt.plot(values, cl.marcenko(values, beta), 'b')

plt.show()