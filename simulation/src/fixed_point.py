
import numpy as np

estimated_pdf = lambda x: 1/np.pi * np.imag(x)

def decontamination(G_k, s, K):
    val = -s
    for k in range(K):
        val += (1 * s * G_k) / (1 - s * np.power(G_k, 2))
    return val

def fixed_point(fixed_point_function, s_vector, iterations, tolerance=.5, attempt_tol=3):
    ret_vec = np.zeros(s_vector.shape)
    epsilons = []
    for ind, s in enumerate(s_vector):
        epsilon = []
        tol_met = False
        attempt = 0
        while not tol_met and attempt < attempt_tol:
            G_k = np.random.uniform(-100, 100) + 1j*1e-6
            for step in range(iterations):
                val = fixed_point_function(s, G_k)
                epsilon.append(np.abs(G_k-val))
                G_k = val
                if epsilon[-1] <= tolerance:
                    ret_vec[ind] = estimated_pdf(G_k)
                    epsilons.append(epsilon[-1])
                    tol_met = True
                    break
            attempt += 1
        if not tol_met:
            print("not converging")
    epsilons = np.asarray(epsilons)
    return ret_vec

