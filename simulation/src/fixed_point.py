
import numpy as np

def estimated_pdf(x):
    return 1/np.pi * np.imag(x)

def decontamination(G_k, s, K):
    val = -s
    for k in range(K):
        val += (1 * s * G_k) / (1 - s * np.power(G_k, 2))
    return val

def fixed_point(fixed_point_function, s_vector, iterations, tolerance=1e-5, attempt_tol=5):
    ret_vec = -1j*1j*np.ones(s_vector.shape)
    epsilons = []
    for ind, s in enumerate(s_vector):
        epsilon = []
        tol_met = False
        attempt = 0
        # Use different initialization points if not converging fast enough
        while not tol_met and attempt < attempt_tol:
            # Fixed point is unique in the upper half complex plane so initialize in this region
            # G_k = np.random.uniform(1, 100) + 1j*np.random.uniform(1, 100)
            G_k = .1
            for step in range(iterations):
                val = fixed_point_function(s, G_k)
                epsilon.append(np.abs(G_k-val))
                G_k = val
                if epsilon[-1] <= tolerance:
                    # ret_vec[ind] = estimated_pdf(G_k)
                    ret_vec[ind] = G_k
                    epsilons.append(epsilon[-1])
                    tol_met = True
                    break
            attempt += 1
        if not tol_met:
            print("not converging")
    epsilons = np.asarray(epsilons)
    return ret_vec

