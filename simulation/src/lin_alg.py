#   About: A collection of functions uses often throughout this work.

import numpy as np
from scipy.stats import bernoulli
from scipy.linalg import sqrtm

def capacity(matrix, snr):
    return np.sum(np.log2(1 + snr*np.linalg.eigvalsh(matrix)))

def capacity_water_filled(matrix):
    return np.sum(np.log2(1 + np.linalg.eigvalsh(matrix)))


def aed_capacity(x_value, value_probability, snr, number_receivers, step=None):
    if step is not None:
        probabilities = value_probability*step
    else:
        probabilities = value_probability
    return number_receivers*np.sum(np.log2(1 + snr*x_value)*probabilities)


def c_bernoulli(rows, cols, p):
    return np.random.randint(0, 2, (rows, cols))


def c_rand(rows, cols, var=None, mean=0):
    if var is not None:
        return (np.random.randn(rows, cols) + 1j*np.random.randn(rows, cols))*np.sqrt(var) + mean
    else:
        return (np.random.randn(rows, cols) + 1j*np.random.randn(rows, cols))/np.sqrt(2*rows) + mean


def block_matrix(block, size, rows=True):
    """
    Create correlation matrices with repeated correlation blocks
    :param block:
    :return:
    """
    dim = size//block.shape[1]
    mat = np.kron(np.eye(dim), block)
    return normalize_matrix(mat, rows=rows)


def exponential_correlation(size, r, rows=True):
    ret_mat = np.ones((size, size))
    for row in range(ret_mat.shape[0]):
        for col in range(ret_mat.shape[1]):
            ret_mat[row, col] *= pow(r, abs(col-row))
    ret_mat = normalize_matrix(sqrtm(ret_mat), rows=rows)
    check = ret_mat@hermetian(ret_mat)
    return ret_mat
    # return normalize_matrix(ret_mat, rows)



def normalize_matrix(matrix, rows=False):
    ret_mat = matrix
    if not rows:
        for col in range(matrix.shape[1]):
            ret_mat[:, col] /= np.linalg.norm(matrix[:, col], 2)
        return ret_mat
    else:
        for row in range(matrix.shape[0]):
            ret_mat[row, :] /= np.linalg.norm(matrix[row, :], 2)
        return ret_mat


def normalize(x):
    normalized = x/np.linalg.norm(x)
    return normalized


def normalize_pdf(x):
    normalized = x / np.sum(x)
    return normalized

c_rand_mean = lambda rows, cols: 1e-1 + 1j*1e-1 + (np.random.randn(rows, cols) + 1j*np.random.randn(rows, cols))/np.sqrt(2*rows)


c_rand_2 = lambda rows, cols:  (np.random.randn(rows, cols) + 1j*np.random.randn(rows, cols))/np.sqrt(cols*rows)

# c_rand = lambda rows, cols: -1j*1j*np.eye(rows)

random_phase = lambda size: np.diag(np.exp(1j * np.random.uniform(0, 2 * np.pi, size)))


hermetian = lambda x: np.conj(x.T)


def reduced_rank(rows, rank):
    los_rank_vector = 1j*np.ones(rows)
    los_rank_vector[int(los_rank_vector.size*rank):los_rank_vector.size] = 0
    return np.diag(los_rank_vector)


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

#Other Useful Functions
pos = lambda x: (x > 0)*x
delta = lambda x: x == 0
