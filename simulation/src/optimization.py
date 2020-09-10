"""
Optimization tools for selecting the phases as IRS elements
"""
import numpy as np
from src.lin_alg import *

def optimize_phases(coefficients, adjacency, F2, F1,  csi = 1, num_repetitions=1, improvements=False):
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
    #   Check how many or the IRS paths are known.

    #   Do not need to encode adjacency matrix since it is structured.
    effective_num_elements = int(adjacency.shape[0]*csi)
    final_phases = -1j*1j*np.ones((adjacency.shape[0]))
    #   First setup binary matrix for easier optimization
    H_opt = F2 @ F1
    beamformer = precode_mf(H_opt)
    progress = [np.trace(-H_opt@beamformer) + np.conjugate(np.trace(-H_opt@beamformer)) + np.trace(H_opt@beamformer@hermetian(H_opt@beamformer))]
    for rep_ind in range(num_repetitions):
        for elem_ind in range(effective_num_elements):
            #   For each phase option choose the one resulting in the largest current sum
            poly_code = adjacency[elem_ind, :]
            off_elements = coefficients * np.logical_not(poly_code)
            best_phase = np.pi - np.angle(poly_code @ coefficients)
            adjustment = np.exp(1j * best_phase) * poly_code * coefficients
            temp_phases = np.copy(final_phases)     # (REQUIRED!) compiler is not copying automatically so forced copy
            temp_phases[elem_ind] *= np.exp(1j * best_phase)
            H_opt = F2 @ np.diag(temp_phases) @ F1
            new = np.trace(-H_opt@beamformer) + np.conjugate(np.trace(-H_opt@beamformer)) + np.trace(H_opt@beamformer@hermetian(H_opt@beamformer))
            if (np.real(new) <= np.real(progress[-1])):
                final_phases[elem_ind] *= np.exp(1j * best_phase)
                coefficients = off_elements + adjustment
                progress.append(new)
    #   Return the final phase offset of each IRS element (i.e. the total phase offset applied during optimization).
    if improvements:
        return np.diag(final_phases), progress
    else:
        return np.diag(final_phases)