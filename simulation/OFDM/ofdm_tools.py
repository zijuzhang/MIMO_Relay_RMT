import numpy as np


def add_cyclic_prefix(symbol_vector, length):
    return np.concatenate((symbol_vector, symbol_vector[:length]), axis=0)

def DFT_matrix(size):
    mat = np.zeros((size,size))
    for i in range(size):
        for j in range(size):
            mat[i,j] = i*j
    return np.exp(-1j*2*np.pi*mat/size)/100
