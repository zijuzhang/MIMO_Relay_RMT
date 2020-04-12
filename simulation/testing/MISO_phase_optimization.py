import numpy as np
import cvxpy as cp
from src.lin_alg import *

H_1 = c_rand(100, 100)
h_2 = c_rand(100, 1)
h = c_rand(100, 1)
f = np.ones((100, 1))

Thetas = cp.Variable((100, 1))
check_thetas = random_phase(100)
#   Note that technically this is the wrong constraint but I am checking if solutions are typically on the perimiter
#   in which case we can relaxy the non-convex constraint.
constraint = [cp.abs(phase) <= 1 for phase in Thetas]
utility = [cp.real(np.conj(h.T)@np.conj(H_1.T)@cp.diag(Thetas).H@h_2 + np.conj(h_2.T)@cp.diag(Thetas)@H_1@h)]
# utility = [cp.norm2((np.conj(h_2.T)@cp.diag(Thetas)@H_1 + np.conj(h.T))@f)]
prob = cp.Problem(cp.Maximize(cp.sum(utility)), constraint)
prob.solve(verbose=True)
Thetas = Thetas.value
opt = prob.solution.opt_val
check = np.conj(h.T)@np.conj(H_1.T)@np.conj(check_thetas.T)@h_2 + np.conj(h_2.T)@check_thetas@H_1@h
constraint_check = np.alltrue(np.abs(Thetas) == 1)
print(opt)
print(check)
print(constraint_check)

check_values = []
for i in range(100):
    check_thetas = random_phase(100)
    check_values.append(hermetian(f)@hermetian(H_1)@hermetian(check_thetas)@h_2@hermetian(h_2)@check_thetas@f)
check_values = np.asarray(check_values)
pass




