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
    rep_val = []
    progress = [np.sum(coefficients) + np.sum(np.conjugate(coefficients))]
    for rep_ind in range(num_repetitions):
        for elem_ind in range(num_elements):
            #   For each phase option choose the one resulting in the largest current sum
            poly_code = adjacency[elem_ind, :]
            off_elements = coefficients * np.logical_not(poly_code)
            best_phase = np.pi - np.angle(poly_code @ coefficients)
            adjustment = np.exp(1j * best_phase) * poly_code * coefficients
            adjust_total = off_elements + adjustment
            new = np.sum(adjust_total) + np.sum(np.conjugate(adjust_total))
            if (new <= progress[-1]):
                final_phases[elem_ind] *= np.exp(1j * best_phase)
                coefficients = off_elements + adjustment
                progress.append(np.sum(coefficients) + np.sum(np.conjugate(coefficients)))
        min_power = np.sum(coefficients) + np.sum(np.conjugate(coefficients))
        rep_val.append(min_power)
    #   Return the final phase offset of each IRS element (i.e. the total phase offset applied during optimization).
    return final_phases