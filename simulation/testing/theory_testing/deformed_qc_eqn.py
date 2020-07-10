from src.stieltjes_eqn import *
import matplotlib.pyplot as plt
from src.circle_laws import *
from src.lin_alg import *
cols = 500
alpha_list = [.1, .5, .8, 1]
bins = 50
los_rank = 1
average = 10
plt.figure()
for alpha in alpha_list:
    x_values, step = np.linspace(1e-2, 10, 500, retstep=True)
    distribution = deformed_qc_eigen(x_values, alpha)
    y = 1e-9
    s_vector = x_values + 1j * y
    check = estimated_pdf(deformed_quarter_circle(s_vector, alpha))
    # plt.plot(x_values, check, label=f"check {alpha}")

    eigen_values = []
    rows = int(cols * alpha)
    for i in range(average):
        H = c_rand(rows, cols, var=1/(2*cols))
        eigen_values.append(np.linalg.eigvalsh(H@hermetian(H)))
    eigen_value_vec, bins = np.histogram(np.asarray(eigen_values), bins=bins)
    plt.plot(bins[:-1], eigen_value_vec / (rows * average), label=f"{alpha}")

plt.title("deformed quarter circle")
# plt.plot(values, st_distribution)
plt.legend(loc="upper right")
plt.show()

