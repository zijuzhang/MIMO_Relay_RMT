import numpy as np

capacity = lambda matrix: np.log(np.linalg.det(np.eye(matrix.shape[0]) + matrix))

c_rand = lambda rows, cols:  (np.random.randn(rows, cols) + 1j*np.random.randn(rows, cols))/np.sqrt(rows)
# c_rand = lambda rows, cols: -1j*1j*np.eye(rows)

random_phase = lambda size: np.diag(np.exp(1j * np.random.uniform(0, 2 * np.pi, size)))

hermetian = lambda x: np.conj(x.T)

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