import cvxpy as cp
from src.lin_alg import *

#   Setup channels

#   Setup system parameters and initialize
size = 100
n_t = 100
n_s = 100
n_r = 100
capacities = []
average = 20

for i in range(average):
    H_1 = c_rand(n_s, n_t, var=1/np.sqrt(n_r*n_s))
    H_2 = c_rand(n_r, n_s, var=1/np.sqrt(n_r*n_r))

    #   Note that perfect CSI at Tx is assumed below

    #   Setup optimization variables
    phases = cp.Variable(n_s)
    powers = cp.Variable(n_t)
    total_channel = H_2@phases@H_1
    power_constraint = 1
    sigma_square = 1
    #   Need to use a closed form of the pseudo-inverse?
    transmit_precoder = hermetian(total_channel)@cp.pinv(total_channel@hermetian(total_channel))
    #   Setup constraints.
    constraints = []
    constraints += [cp.trace(hermetian(transmit_precoder)@cp.diag(powers)@transmit_precoder) <= power_constraint]
    constraints += [0 <= cp.abs(phases[ind]) <= 1 for ind in range(powers.size)]
    # constraints += [cp.abs(phases[ind]) <= 1 for ind in range(powers.size)]
    utility = []
    utility += [cp.log(1+powers[ind]/sigma_square) for ind in range(powers.size)]
    prob = cp.Problem(cp.Maximize(cp.sum(utility)), constraints)
    prob.solve()
    capacities.append(prob.solution)
    capacities.append(capacity_water_filled(total_channel@covariance_matrix@hermetian(total_channel)))
    #   Check irs magnitudes

print(np.average(capacities))


