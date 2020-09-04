"""
Optimization tools for selecting the phases as IRS elements
"""
import numpy as np

def optimize_phases(coefficients, adjacency, num_repetitions=1):
    """
    Given a sum of complex numbers rotated by a number of phases with overlap, this is a sub-optimal algorithm for
    finding a set of phases to minimize this sum. Note that the complex numbers are implied to be included in a
    sum of the numbers (with phase offsets) and the complex conjugates of the nubmers. With that in mind this
    always minimizes the real component of the sum in each step since the complex components would cancel anyways.
    :param coefficients:
    :param adjacency:
    :param num_repetitions:
    :return:
    """
    #   Do not need to encode adjacency matrix since it is structured.
    num_elements = adjacency.shape[0]
    final_phases = -1j*1j*np.ones((adjacency.shape[0]))
    #   First setup binary matrix for easier optimization
    original = np.sum(coefficients) + np.sum(np.conjugate(coefficients))
    rep_val = []
    progress = []
    for rep_ind in range(num_repetitions):
        for elem_ind in range(num_elements):
            #   For each phase option choose the one resulting in the largest current sum
            progress.append(np.sum(coefficients) + np.sum(np.conjugate(coefficients)))
            poly_code = adjacency[elem_ind, :]
            off_elements = coefficients * np.logical_not(poly_code)
            best_phase = np.pi - np.angle(poly_code @ coefficients)
            adjustment = np.exp(1j * best_phase) * poly_code * coefficients
            adjust_total = off_elements + adjustment
            new = np.sum(adjust_total) + np.sum(np.conjugate(adjust_total))
            if (new <= progress[-1]):
                final_phases[elem_ind] *= np.exp(1j * best_phase)
                coefficients = off_elements + adjustment
        min_power = np.sum(coefficients) + np.sum(np.conjugate(coefficients))
        rep_val.append(min_power)
    #   Return the final phase offset of each IRS element (i.e. the total phase offset applied during optimization).
    return final_phases

def optimize_phases_max(coefficients, adjacency, num_repetitions=1):
    """
    Variant of above to maximize complex component
    Given a sum of complex numbers rotated by a number of phases with overlap, this is a sub-optimal algorithm for
    finding a set of phases to minimize this sum.
    :param coefficients:
    :param adjacency:
    :param num_repetitions:
    :return:
    """
    #   Do not need to encode adjacency matrix since it is structured.
    num_elements = adjacency.shape[0]
    final_phases = 1j*np.ones((adjacency.shape[0]))
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
            # best_phase = np.angle(total_off) + np.pi - np.angle(poly_code @ coefficients)
            best_phase = np.pi/2 - np.angle(poly_code @ coefficients)
            final_phases[elem_ind] *= np.exp(1j * best_phase)
            adjustment = np.exp(1j * best_phase) * poly_code * coefficients
            coefficients = off_elements + adjustment
            min_power = np.abs(np.sum(coefficients))
        rep_val.append(min_power)
    gain = original - rep_val
    #   Return the final phase offset of each IRS element (i.e. the total phase offset applied during optimization).
    return final_phases