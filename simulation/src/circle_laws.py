import numpy as np
from src.fixed_point import *
#Other Useful Functions
pos = lambda x: (x > 0)*x
delta = lambda x: x == 0

#General Definitions
estimated_pdf = lambda x: 1/np.pi * np.imag(x)

def normalize(x):
    normalized = x/np.linalg.norm(x)
    check = np.linalg.norm(normalized)
    return normalized



# Quarter Circle Related
st_quarter_circle_eigen = lambda s: 1/2 * np.sqrt(1-4/s) - 1/2
st_quarter_circle = lambda s: 2*np.sqrt(4-np.power(s, 2))/np.pi * np.log((2+np.sqrt(4-np.power(s, 2)))/(-s))-s/2-2/np.pi
marcenko_st2 = lambda s, phi: np.sqrt(np.power((1-phi), 2)/(4*np.power(s, 4)) - (1+phi)/(2*np.power(s, 2))
                   + 1/4) - 1/2 - (1-phi)/(2*np.power(s, 2))

marcenko_st = lambda s, phi: 1/2 + (1-phi)/(2*s) + np.sqrt(np.power(1-phi,2)/(4*np.power(s,2))-(1+phi)/(2*s)+1/4)

#Semi-Circle law related equations

def st_half_circle(s):
    return (s / 2) * np.sqrt(1 - 4 / np.power(s, 2)) - s / 2

def semi_cicle_aed(x):
    return 1 / (2 * np.pi)*np.sqrt(4-np.power(x, 2))


def half_circle_gamma_s_recip_wrong(s, gamma_z_recip_s):
    check = np.sqrt(gamma_z_recip_s)
    return s*(np.sqrt(gamma_z_recip_s) - 1/s)

def half_circle_gamma_s_recip(s, gamma_z_recip_s):
    return 1/(s*(np.power(gamma_z_recip_s, -1/2) - 1/s))

def half_circle_stieltjes_gamma(s_vector, iterations):
    return -(1/s_vector)*(fixed_point(half_circle_gamma_s_recip, s_vector, iterations)+1)


def deformed_qc_eigen(x, alpha):
    #TODO don't allow nan
    interval = np.logical_and(np.power(1-np.sqrt(alpha), 2) < x, x < np.power(1+np.sqrt(alpha), 2))
    output = interval*np.sqrt((x-np.power(1-np.sqrt(alpha), 2))*(np.power(1+np.sqrt(alpha), 2)-x))/(2*np.pi*alpha*x)
    nan = np.isnan(output)
    output[nan] = 0
    # check = np.logical_not(interval)*(1-1/alpha)*(alpha > 1)
    output += np.logical_not(interval)*(1-1/alpha)*(alpha > 1)
    return output


def eigen_vector_compare(e_values, e_vectors):
    """
    assuming that eigenvectors are provided as columns (as is standard for the numpy EVD)
    :param e_values:
    :param e_vectors:
    :return:
    """
    e_value_sums = []
    for ind in range(e_vectors.shape[1]):

        e_vector = e_vectors[:, ind]
        e_value = e_values[ind]
        max_correlation = 0
        for ind_comp in range(e_vectors.shape[1]):
            if ind_comp != ind:
                e_vector_comp = e_vectors[:, ind_comp]
                correlation = e_vector.T@e_vector_comp
                if correlation > max_correlation:
                    e_value_comp = e_values[ind_comp]
                    e_value_sum = np.abs(e_value_comp + e_value)
        e_value_sums.append(e_value_sum)
    return np.asarray(e_value_sums)


def eigen_vector_compare_2(e_values, e_vectors):
    """
    assuming that eigenvectors are provided as columns (as is standard for the numpy EVD)
    :param e_values:
    :param e_vectors:
    :return:
    """
    e_value_sums = 1j*np.zeros(e_values.shape[0])
    for ind in range(e_vectors.shape[1]):
        e_value_sums += e_values[ind]*e_vectors[:, ind]
    return np.abs(e_value_sums)


