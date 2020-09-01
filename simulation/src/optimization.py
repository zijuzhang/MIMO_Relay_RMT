"""
Optimization tools for selecting the phases as IRS elements
"""
import numpy as np

def optimize_phases(coefficients, adjacency, num_repetitions=1):
    #   Do not need to encode adjacency matrix since it is structured.
    num_elements = adjacency.shape[0]
    final_phases = np.ones((adjacency.shape[0]))
    #   First setup binary matrix for easier optimization
    original = np.abs(np.sum(coefficients))
    rep_val = []
    for rep_ind in range(num_repetitions):
        min_power = 0
        for elem_ind in range(num_elements):
            #   For each phase option choose the one resulting in the largest current sum
            best_phase_ind = 0
            min_power = np.inf
            poly_code = adjacency[elem_ind, :]
            off_elements = coefficients * np.logical_not(poly_code)
            total_off = np.sum(off_elements)
            # for phase_ind in range(num_distrete_vals):
            #     phase_code = np.exp(1j*phase_ind*(np.pi/2))*poly_code
            #     cur = np.abs(phase_code@complex_coefficient + total_off)
            #     if cur < min_power:
            #         min_power = cur
            #         best_phase_ind = phase_ind
            # adjustment = np.exp(1j*best_phase_ind*(np.pi/2))*poly_code*complex_coefficient
            check = np.angle(poly_code @ coefficients)
            best_phase = np.angle(total_off) + np.pi - np.angle(poly_code @ coefficients)
            final_phases[elem_ind] *= np.exp(1j * best_phase)
            adjustment = np.exp(1j * best_phase) * poly_code * coefficients
            coefficients = off_elements + adjustment
            min_power = np.abs(np.sum(coefficients))
        rep_val.append(min_power)
    gain = original - rep_val
    #   Return the final phase offset of each IRS element (i.e. the total phase offset applied during optimization).
    return final_phases