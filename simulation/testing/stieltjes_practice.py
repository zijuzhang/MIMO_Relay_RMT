import numpy as np
import matplotlib.pyplot as plt
from src import circle_laws as cl
from src.fixed_point import *

y = 1e-9
values = np.linspace(0.1, 20, 200) + 1j*y
evaluated_fixed_point, epsilons = fixed_point(decontamination, values, 10, 50)
plt.plot(values, evaluated_fixed_point)
evaluated_fixed_point, epsilons = fixed_point(decontamination, values, 10, 50)
plt.plot(values, evaluated_fixed_point)
# plt.plot(values, cl.estimated_pdf(values),'r')
plt.show()
