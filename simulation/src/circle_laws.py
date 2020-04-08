import numpy as np

pos = lambda x: (x > 0)*x
delta = lambda x: x == 0
semi_cicle_aed = lambda x: (1/2*np.pi)*np.sqrt(4-np.power(x,2))
st_half_circle = lambda s: (s / 2) * np.sqrt(1 - 4 / np.power(s, 2)) - s / 2
st_quarter_circle = lambda s: 2*np.sqrt(4-np.power(s, 2))/np.pi * np.log((2+np.sqrt(4-np.power(s, 2)))/(-s))-s/2-2/np.pi
estimated_pdf = lambda x: 1/np.pi * np.imag(x)
true_pdf = lambda x: 1 / (2 * np.pi)*np.sqrt(4-np.power(x, 2))
marcenko = lambda x, beta: pos(1-1/beta)*delta(x) + \
                           np.sqrt(pos(x-np.power((1-np.sqrt(beta)), 2))*pos(np.power((1+np.sqrt(beta)), 2)-x)
                                   / (2*np.pi*beta*x))


def deformed_qc_eigen(x, alpha):
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


