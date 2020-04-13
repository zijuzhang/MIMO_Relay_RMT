
import numpy as np

estimated_pdf = lambda x: 1/np.pi * np.imag(x)


def decontamination_function(s_vector, K, iterations, tolerance=1e-1, attempt_tol=1):
    ret_vec = np.zeros(s_vector.shape)
    epsilons = []
    for ind, s in enumerate(s_vector):
        epsilon = []
        tol_met = False
        attempt = 0
        while not tol_met and attempt < attempt_tol:
            G_k = np.random.uniform(-100, 100)
            for step in range(iterations):
                val = -s
                for k in range(K):
                    val += (1*s*G_k)/(1-s*np.power(G_k, 2))
                epsilon.append(np.abs(G_k-val))
                G_k = val
                if epsilon[-1] <= tolerance:
                    ret_vec[ind] = estimated_pdf(G_k)
                    epsilons.append(epsilon[-1])
                    tol_met = True
                    break
        if not tol_met:
            print("not converging")
    epsilons = np.asarray(epsilons)
    return ret_vec, epsilons

