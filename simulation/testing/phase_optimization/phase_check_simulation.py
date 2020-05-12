# About:
#   This is a test to look at how much impact different selections for the phase values at the IRS will
#   have on the capacity of the channel.
# Takeaways:
#   As expected from analytic results, the phases will have no impact on the capacity for the asymptotic limit.
#   In contrast, there is significant impact when there are a low number of transmit or receive antennas (including
#   the case in which only one of these dimensions is small)


from src.lin_alg import *

size = 100
n_t = 3
n_r = 3
rows = size
cols = size
bins = int(size/2)


capacities = []
cross_sum = 0
main_sum = 0

average = 1000
# H_1 = c_rand(size, n_t)
H_1 = c_rand(size, n_t, var=1/np.sqrt(n_r*size))
H_2 = c_rand(n_r, size, var=1/np.sqrt(n_r*size))
# H_1 = c_rand(size, n_t)
# H_2 = c_rand(n_r, size)
G = c_rand(n_r, n_t)
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
