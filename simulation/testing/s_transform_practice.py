import numpy as np
import matplotlib.pyplot as plt
from src.circle_laws import *
from src import circle_laws as cl
from src.fixed_point import *

y = 1e-9
s_values = np.linspace(-2, 2, 100) + 1j*y
x_values = np.linspace(-2, 2, 100)

plt.plot(x_values, estimated_pdf(st_half_circle(s_values)))
plt.plot(x_values, estimated_pdf(half_circle_stieltjes_gamma(s_values, 20)))
plt.plot(x_values, semi_cicle_aed(x_values))

plt.show()
