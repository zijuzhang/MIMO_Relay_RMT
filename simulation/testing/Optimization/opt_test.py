import numpy as np
from src.sum_optimization import *
from src.lin_alg import *
num_elements = 20
num_distrete_vals = 4
num_repetitions = 2
rep_val = []
complex_coefficient = np.random.uniform(-10, 10, 100) + 1j * np.random.uniform(-10, 10, 100)
phase_adjacency = np.random.randint(0, 100, (num_elements, 10))
coded = np.zeros((num_elements, complex_coefficient.size))
for i in range(phase_adjacency.shape[0]):
    for j in range(phase_adjacency.shape[1]):
        coded[i, phase_adjacency[i, j]] = 1
# coded = np.concatenate((coded, coded), axis=1)
# complex_coefficient = np.concatenate((complex_coefficient, np.conjugate(complex_coefficient)))
cur_sum = 0
min_power = np.inf
original = np.abs(np.sum(complex_coefficient))
check = []
iterations_skipped = 0
for rep_ind in range(num_repetitions):
    for elem_ind in range(num_elements):
        check.append(np.sum(complex_coefficient)+np.sum(np.conjugate(complex_coefficient)))
        #   For each phase option choose the one resulting in the largest current sum
        # min_power = np.inf
        poly_code = coded[elem_ind, :]
        off_elements = complex_coefficient*np.logical_not(poly_code)
        one = np.sum(poly_code@complex_coefficient + np.conjugate(poly_code@complex_coefficient))
        best_phase = np.pi - np.angle(poly_code@complex_coefficient)
        adjustment = np.exp(1j*best_phase)*poly_code*complex_coefficient
        two = np.sum(adjustment + np.conjugate(adjustment))
        adjust_total = off_elements + adjustment
        new = np.sum(adjust_total)+np.sum(np.conjugate(adjust_total))
        if (new <= check[-1]):
            complex_coefficient = off_elements + adjustment
        else:
            iterations_skipped +=1
        # complex_coefficient = off_elements + adjustment
    rep_val.append(np.sum(complex_coefficient)+np.sum(np.conjugate(complex_coefficient)))
check1 = np.linalg.norm(rep_val[-1] - original)/np.linalg.norm(original)
improvement = 20*np.log(np.linalg.norm(rep_val[-1] - original)/np.linalg.norm(original))
print(f"improvement: {improvement}")
print(f"skipped iterations: {iterations_skipped}")