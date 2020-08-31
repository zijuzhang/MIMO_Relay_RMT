import numpy as np
from src.sum_optimization import *
from src.lin_alg import *
num_elements = 100
num_distrete_vals = 4
num_repetitions = 2
rep_val = []
complex_coefficient = np.random.uniform(-10, 10, 100) + 1j * np.random.uniform(-10, 10, 100)
phase_adjacency = np.random.randint(0, 100, (num_elements, 50))
coded = np.zeros((num_elements, complex_coefficient.size))
for i in range(phase_adjacency.shape[0]):
    for j in range(phase_adjacency.shape[1]):
        coded[i, phase_adjacency[i, j]] = 1
cur_sum = 0
min_power = np.inf
original = np.abs(np.sum(complex_coefficient))
for rep_ind in range(num_repetitions):
    for elem_ind in range(num_elements):
        #   For each phase option choose the one resulting in the largest current sum
        best_phase_ind = 0
        min_power = np.inf
        poly_code = coded[elem_ind, :]
        off_elements = complex_coefficient*np.logical_not(poly_code)
        total_off = np.sum(off_elements)
        # for phase_ind in range(num_distrete_vals):
        #     phase_code = np.exp(1j*phase_ind*(np.pi/2))*poly_code
        #     cur = np.abs(phase_code@complex_coefficient + total_off)
        #     if cur < min_power:
        #         min_power = cur
        #         best_phase_ind = phase_ind
        # adjustment = np.exp(1j*best_phase_ind*(np.pi/2))*poly_code*complex_coefficient
        check = np.angle(poly_code@complex_coefficient)
        best_phase = np.angle(total_off) + np.pi - np.angle(poly_code@complex_coefficient)
        adjustment = np.exp(1j*best_phase)*poly_code*complex_coefficient
        complex_coefficient = off_elements + adjustment
        min_power = np.abs(np.sum(complex_coefficient))
    rep_val.append(min_power)
gain = original - rep_val
print(gain)
#   Adjust the complex coefficients using the winning phase