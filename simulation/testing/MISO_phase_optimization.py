import numpy as np
import cvxpy as cp
from src.lin_alg import *
size = 1000
H_1 = c_rand(size, size)
h_2 = c_rand(size, 1)
h = c_rand(size, 1)
f = np.ones((size, 1))

check_values = []
for i in range(50):
    check_thetas = random_phase(size)
    # check_values.append(hermetian(f)@hermetian(H_1)@hermetian(check_thetas)@h_2@hermetian(h_2)@check_thetas@f)
    check_values.append(np.linalg.norm(hermetian(h_2)@check_thetas@H_1@f, 2))
    pass
check_values = np.asarray(check_values)


Thetas = cp.Variable((size, 1))
check_thetas = random_phase(size)
#   Note that technically this is the wrong constraint but I am checking if solutions are typically on the perimiter
#   in which case we can relaxy the non-convex constraint.
constraint = [cp.abs(phase) <= 1 for phase in Thetas]
utility = [cp.real(np.conj(h.T)@np.conj(H_1.T)@cp.diag(Thetas).H@h_2 + np.conj(h_2.T)@cp.diag(Thetas)@H_1@h)]
prob = cp.Problem(cp.Maximize(cp.sum(utility)), constraint)
prob.solve(verbose=True)
Thetas = Thetas.value
opt = prob.solution.opt_val
check = np.conj(h.T)@np.conj(H_1.T)@np.conj(check_thetas.T)@h_2 + np.conj(h_2.T)@check_thetas@H_1@h
constraint_check = np.alltrue(np.abs(Thetas) == 1)
main_term_check = np.linalg.norm(hermetian(h_2)@np.diag(Thetas.flatten())@H_1@f, 2)
print(opt)
print(check)
print(constraint_check)

pass




