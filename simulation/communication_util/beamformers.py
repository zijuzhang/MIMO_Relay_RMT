import cvxpy as cp
from src.lin_alg import *

def water_filling(channel_matrix, power_constraint, sigma_square=1, vals=False):
    """
    Use of CVX to solve the convex water-filling problem for power allocation in a MIMO channel with CSI at
    the transmitter.
    :param covariance_matrix:
    :param power_constraint:
    :param sigma_square:
    :return:
    """
    U, Sigma, V_H = np.linalg.svd(channel_matrix)
    variables = cp.Variable(channel_matrix.shape[1])
    constraint = [cp.sum(variables) <= power_constraint]
    constraint += [variables >= 0]
    # utility = cp.sum(cp.log(1+variables*e_values/sigma_square))
    utility = []
    utility += [cp.log(1+variables[ind]*Sigma[ind]/sigma_square) for ind in range(variables.size)]
    prob = cp.Problem(cp.Maximize(cp.sum(utility)), constraint)
    prob.solve()
    covariance_matrix = hermetian(V_H)@np.diag(variables.value)@V_H
    check = np.trace(covariance_matrix)
    if vals:
        return np.diag(variables.value)
    else:
        return channel_matrix@covariance_matrix@hermetian(channel_matrix)


def maximum_ratio_transmission(CSI):
    # This normalization ensures the power of the transmit signal signal is unaffected.
    beamformer = np.conjugate(CSI/np.linalg.norm(CSI))
    return beamformer