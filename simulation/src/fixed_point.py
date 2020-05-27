
import numpy as np


def decontamination(G_k, s, K):
    val = -s
    for k in range(K):
        val += (1 * s * G_k) / (1 - s * np.power(G_k, 2))
    return val


def fixed_point(fixed_point_function, s_vector, iterations,
                tolerance=1e-1, attempt_tol=1, func_param=None, unique_half_plane=False):
    """
    NOTE that I do not do perform the function concurrently. This allows for better
    tracking of the fixed-point convergence.
    equation
    :param fixed_point_function:
    :param s_vector:
    :param iterations:
    :param tolerance:
    :param attempt_tol:
    :param unique_half_plane: Inidicates if the fixed point is only unique in the upper half plane.
    :return:
    """
    if type(s_vector) is not np.ndarray:
        ret_vec = -1j*1j*np.zeros((1, 1))
        s_vector = [s_vector]
    else:
        ret_vec = -1j*1j*np.zeros(s_vector.shape)
    epsilons = []
    for ind, s in enumerate(s_vector):
        epsilon = []
        tol_met = False
        attempt = 0
        check = []
        # non_negative = True
        # Use different initialization points if not converging fast enough
        while not tol_met and attempt < attempt_tol:
            # Fixed point is unique in the upper half complex plane so initialize in this region
            G_k = np.random.uniform(0, 1) + 1j*np.random.uniform(0, 1)
            G_k = G_k / np.linalg.norm(G_k) # Check to see if this is needed for convergenece
            # G_k = .1
            for step in range(iterations):
                if func_param is not None:
                    val = fixed_point_function(s, G_k, func_param)
                else:
                    val = fixed_point_function(s, G_k)
                if unique_half_plane and np.imag(val) < 0:
                    val = np.conj(val)
                if np.isinf(val) or np.isinf(val) or np.isclose(0, val):
                    print("nan/inf")
                epsilon.append(np.abs(G_k - val))
                G_k = val
                check.append(val)
                ret_vec[ind] = G_k
                if epsilon[-1] <= tolerance:
                    epsilons.append(epsilon[-1])
                    tol_met = True
                    # break
            check1 = np.asarray(check)
            attempt += 1
        if tol_met:
            check1 = np.asarray(check)
        if not tol_met:
            print("not converging")
    epsilons = np.asarray(epsilons)
    return ret_vec

