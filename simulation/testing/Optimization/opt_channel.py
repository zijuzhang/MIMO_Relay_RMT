import numpy as np
from src.lin_alg import *
from src.optimization import *


average = 100
num_tx = 16
num_rx = 10
num_reflectors = 50

for j in range(average):
    #   Each IRS path is a column of F2 @ a row of F1
    F1 = c_rand(num_reflectors, num_tx, var=1)
    F2 = c_rand(num_rx, num_reflectors, var=1)
    G = c_rand(num_rx, num_tx, var=1)
    coefficients = []
    for elem_ind in range(num_reflectors):
        ind = elem_ind * num_reflectors
        for elem2_ind in range(num_reflectors - elem_ind):
            elem2_ind += elem_ind
            f2i = F2[:, elem_ind]
            f1i = F1[elem_ind, :]
            f2j = F2[:, elem2_ind]
            f1j = F1[elem2_ind, :]
            coefficient = f1i.T@np.conjugate(f1j)*hermetian(f2j)@f2i
            coefficients.append(coefficient)

    phase_adjacency = np.zeros((num_reflectors, len(coefficients)))
    ind = 0
    for elem_ind in range(num_reflectors):
        for elem2_ind in range(num_reflectors - elem_ind):
            phase_adjacency[elem_ind, ind + elem2_ind] = 1
            phase_adjacency[elem2_ind, ind + elem2_ind] = 1
        ind += num_reflectors - elem_ind
    optimal_phases = optimize_phases(np.asarray(coefficients), phase_adjacency)
    pass


