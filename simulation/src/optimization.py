"""
Optimization tools for selecting the phases as IRS elements
"""
import numpy as np
from src.lin_alg import *

def optimize_phases_general():
    """

    :return:
    """
    return None

def optimize_phases_preselected(coefficients, adjacency, F2, F1,  csi = 1, num_repetitions=1, improvements=False,
                                maximize=False ):
    """
    Only use for pre-selected beamformer!
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
            polynomial_code = adjacency[elem_ind, :]
            # Why do I take logical not below? -> because these are the elements not impact by the phase
            off_elements = coefficients * np.logical_not(polynomial_code)
            if not maximize:
                # In this case, I try to make the contribution negative and real.
                best_phase = np.pi - np.angle(polynomial_code @ coefficients)
            else:
                # In this case,  I try to make the contribution positive and real.
                best_phase = 2*np.pi - np.angle(polynomial_code @ coefficients)
            adjustment = np.exp(1j * best_phase) * polynomial_code * coefficients
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


def get_polynomial_coefficients(F1, F2, beamformer, num_reflectors):
    """
    Returns a list of coefficients for the phase optimization of an IRS in between matrix channels.
    :param F1:
    :param F2:
    :param beamformer:
    :param num_reflectors:
    :return:
    """
    coefficients = []
    non_cross = -1j*1j*0
    for elem_ind in range(num_reflectors):
        #   Pick out rows and columns of the channels corresponding to specific IRS elements
        f2i = F2[:, elem_ind]
        f1i = F1[elem_ind, :]
        i_mat = np.outer(f2i, f1i)
        non_cross += np.trace(i_mat@beamformer@hermetian(i_mat@beamformer))
        # Add all of the cross components of the polynomial to the list of coefficients.
        for elem2_ind in range(num_reflectors - elem_ind - 1):
            elem2_ind += elem_ind + 1
            f2j = F2[:, elem2_ind]
            f1j = F1[elem2_ind, :]
            j_mat = np.outer(f2j, f1j)
            # Below wraps around to get complex scalar by the trace rotation property.
            coefficient = f1i.T@beamformer@hermetian(beamformer)@np.conjugate(f1j)*hermetian(f2j)@f2i
            # coefficient = f1i.T@np.conjugate(f1j)*hermetian(f2j)@f2i
            # coefficient1 = np.trace(i_mat@transmit_matched_rand@hermetian(transmit_matched_rand)@hermetian(j_mat))
            coefficients.append(coefficient)
    # Create matrix to pick out the MSE terms influenced by specific IRS elements.
    for elem_ind in range(num_reflectors):
        f2i = F2[:, elem_ind]
        f1i = F1[elem_ind, :]
        # Is this coming from the MMSE term?
        coefficients.append(-np.trace(np.outer(f2i, f1i)@beamformer))
        # Non-cross terms are not important for optimization but keep track of the total term.
    return coefficients

def get_polynomial_adjacency_matrix(num_reflectors, num_coeffifients):
    """
    Return the matrix to pick out values of a vector corresponding to a particular IRS
    :return: 
    """
    ind = 0
    phase_adjacency = np.concatenate((np.zeros((num_reflectors, num_coeffifients)), np.eye(num_reflectors)), axis=1)
    for elem_ind in range(num_reflectors-1):
        for elem2_ind in range(num_reflectors - elem_ind - 1):
            phase_adjacency[elem_ind, ind + elem2_ind] = 1
            elem3_ind = elem2_ind + elem_ind + 1
            phase_adjacency[elem3_ind, ind + elem2_ind] = 1
        ind += num_reflectors - elem_ind - 1
    return phase_adjacency


def optimize_MISO_phase(LOS_Channel, IRS_coefficients, iteration_rounds=1, relative_pilots=1):
    phase = -1j*1j*np.ones((1, IRS_coefficients.size))
    cur_sum  = LOS_Channel
    for round in range(iteration_rounds):
        for i in range(int(np.round(phase.size*relative_pilots))): # Only optimize for number of pilots sent
            phase[0, i] = np.exp(1j*(np.angle(cur_sum) - np.angle(IRS_coefficients[0, i])))
            cur_sum += phase[0, i]*IRS_coefficients[0, i]
    return phase.flatten()