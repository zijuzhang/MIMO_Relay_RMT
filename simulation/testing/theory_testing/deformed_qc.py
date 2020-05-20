from src.stieltjes_eqn import *
import matplotlib.pyplot as plt
from src.circle_laws import *
from src.lin_alg import *
rows = 100
# alpha_list = [.1, .5, .8, 1]
alpha_list = [.4]
# alpha_list = [1]
bins = 50
los_rank = 1
average = 10
plt.figure()
for alpha in alpha_list:
    eigen_values = []
    cols = int(rows * alpha)
    for i in range(average):
        los_rank_matrix = reduced_rank(rows, los_rank)
        H_los = c_rand(rows, cols, 1/(2*rows), mean=1j)
        total = H_los@hermetian(H_los)
        # svd = np.linalg.svd(H_los)
        # eigen_value, bins = np.histogram(svd[1], bins=bins)
        trc = np.trace(total)
        check = np.linalg.eigh(total)
        e_vec = check[1][:, -1]
        check1 = np.linalg.norm(e_vec)
        eigen_values.append(np.linalg.eigvalsh(total))

    eigen_value_vec, bins = np.histogram(np.asarray(eigen_values), bins=bins)
    plt.plot(bins[:-1], eigen_value_vec/(rows*average), label=f"{alpha}")

    x_values, step = np.linspace(1e-2, 10, 100, retstep=True)
    distribution = deformed_qc_eigen(x_values, alpha)

    y = 1e-9
    s_vector = x_values + 1j * y

    st_vals = st_quarter_gamma_based(s_vector, alpha)
    second_distribution = estimated_pdf(st_vals)

    st_vals = st_dquarter_gamma_based(s_vector, alpha)
    third_distribution = estimated_pdf(st_vals)

    sigma = 1
    rho = 1
    phi = 1
    parameters = [sigma, rho, phi]
    theoretic_pdf = estimated_pdf(fixed_point(final_stieltjes_deformed, s_vector, 100, 1e-3, 3,
                                              func_param=parameters, unique_half_plane=True))


    # plt.plot(x_values, distribution, label=f"true {alpha}")
    # plt.plot(x_values, second_distribution, label=f"gamma qc {alpha}")
    # plt.plot(x_values, third_distribution, label=f"gamma dqc {alpha}")
    plt.plot(x_values, theoretic_pdf, label=f"conc dqc {alpha}")


# st_distribution = estimated_pdf(st_quarter_circle_eigen(values+1j*1e-6))
print(f" simulation capacity: {capacity(total,1/cols)}")
# print(f"theoretic capacity: {aed_capacity(values, st_distribution, step, 1/cols, rows)}")
plt.title("deformed quarter circle")
# plt.plot(values, st_distribution)
plt.legend(loc="upper right")
plt.show()

