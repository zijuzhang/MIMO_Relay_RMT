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

