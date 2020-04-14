from src.circle_laws import *
from src.lin_alg import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn
size = 500
bins = int(size/2)
bins = 50
alpha = 1
H_1 = c_rand(size, size)
H_2 = c_rand(size, alpha*size)

s_vals = np.linalg.svd(H_2)[1]
Symmetric_ASD, asd_bins = np.histogram(np.asarray(s_vals), bins=bins)

H_2 = H_2@hermetian(H_2)

e_vals = np.linalg.eig(H_2)[0]
Symmetric_AED, aed_bins = np.histogram(np.asarray(e_vals), bins=bins)


fig, ax = plt.subplots()
plt.title("AED vs. ASD")

ax.plot(asd_bins[1::], Symmetric_ASD/size, label='ASD')

ax.plot(aed_bins[1::], Symmetric_AED/size, label='AED')
plt.show()