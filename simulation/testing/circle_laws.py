import numpy as np

pos = lambda x: (x > 0)*x
delta = lambda x: x == 0
st_half_circle = lambda s: (s / 2) * np.sqrt(1 - 4 / np.power(s, 2)) - s / 2
st_quarter_circle = lambda s: 2*np.sqrt(4-np.power(s, 2))/np.pi * np.log((2+np.sqrt(4-np.power(s, 2)))/(-s))-s/2-2/np.pi
estimated_pdf = lambda x: 1/np.pi * np.imag(x)
true_pdf = lambda x: 1 / (2 * np.pi)*np.sqrt(4-np.power(x, 2))
marcenko = lambda x, beta: pos(1-beta)*delta(x) + \
                           np.sqrt(pos(x-np.power((1-np.sqrt(beta)), 2))*pos(np.power((1+np.sqrt(beta)), 2)-x)
                                   / (2*np.pi*beta*x))

c_rand = lambda rows, cols:  np.random.randn(rows, cols)/np.sqrt(rows) + 1j*np.random.randn(rows, cols)/np.sqrt(rows)

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


def is_orthonormal(matrix):
    orthonormal = []
    for ind in range(matrix.shape[1]):
        e_vector = matrix[:, ind]
        unit_modulus = np.linalg.norm(e_vector) == 1
        orthogonal = []
        for ind_comp in range(matrix.shape[1]):
            e_vector_comp = matrix[:, ind_comp]
            if ind_comp != ind:
                correlation = e_vector.T@e_vector_comp
                if correlation != 0:
                    orthogonal.append(False)
                else:
                    orthogonal.append(True)
        if np.alltrue(np.asarray(orthogonal)) and unit_modulus:
            return True
        else:
            return False

    return np.alltrue(np.asarray(orthonormal))