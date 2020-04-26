from src.circle_laws import *
import numpy as np
import matplotlib.pyplot as plt
from src.lin_alg import *
import seaborn

size = 100
n_t = 4
n_r = 4
rows = size
cols = size
bins = int(size/2)


capacities = []
cross_sum = 0
main_sum = 0

average = 50
H_1 = c_rand(size, n_t)
H_2 = c_rand(n_r, size)
G = c_rand(n_r, n_t)
F = c_rand(rows, cols)
for i in range(average):
    H = H_2 @ np.diag(np.exp(1j * np.random.uniform(0, 2 * np.pi, size))) @ H_1
    # F = c_rand(rows, cols) @ np.diag(np.exp(1j * np.random.uniform(0, 2 * np.pi, rows))) @ c_rand(rows, cols)
    GG = G @ np.conj(G.T)
    HH = H @ np.conj(H.T)
    GH = G @ np.conj(H.T)
    HG = H @ np.conj(G.T)
    cross = GH + HG
    main = HH + GG
    total_irs = cross + main
    # total_irs = GG
    capacities.append(capacity(total_irs, 1/n_t))

print(f"Capacity variance:\n {np.var(capacities)}")
print(f"Capacity average:\n {np.average(capacities)}")
print(f"Capacity min:\n {np.min(capacities)}")
print(f"Capacity max:\n {np.max(capacities)}")
print(f"Capacities:\n {capacities}")
