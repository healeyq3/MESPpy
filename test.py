import pandas as pd
import numpy as np
from numpy.linalg import slogdet
import os

from mesp import Mesp
from mesp.utilities import matrix_computations
from mesp.branching.variable_fixing import varfix

C = pd.read_table(os.getcwd() + "/examples/data/Data63.dms",
                  header=None, encoding="utf-8", sep="\s+")

n = 63
C = np.array(C)
C = C.reshape(n,n)

C = 100*np.matrix(C)
# V, Vsquare, E = matrix_computations.generate_factorizations(C, n, n)
# print("V Square type", type(Vsquare[0]))
# S1, S0, time = varfix(V, Vsquare, E, n, n, 10)
# print(S1, S0, time)
# print("V shape", V.shape)

test_mesp = Mesp(C)

# z_hat = test_mesp.solve_approximate(s=20)
# print(z_hat)
# succ_fix, C_hat, n_hat, d_hat, s_hat, scale = test_mesp.fix_variables(s=20)
# print(scale)
# print(n_hat, s_hat)
# z_hat = test_mesp.solve_approximate(s=s_hat)[0]
# print(z_hat)

# succ_fix, C_hat, n_hat, d_hat, s_hat, scale = test_mesp.fix_variables(5)
solved, opt_val, time, iterations, gap, z_hat, num_updates = test_mesp.solve(50, fix_vars=True)
print(f"solved: {solved}")
print(f"opt_val: {opt_val}")
print(f"iterations: {iterations}")
print(f"gap: {gap}")
print(f"z_hat: {z_hat}")
print(f"num_updates: {num_updates}")
# print(f"succ fix {succ_fix}")
# print(f"s_hat")
# print(f"scale {scale}")
# print(matrix_computations.obj_f([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1], Vsquare))
# print(test_mesp.solve_approximate(10)[0])

# # indices = np.flatnonzero([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1])
# # print(indices)

# # print(slogdet(C[[1]][:, [1]])[1])
# # print(np.linalg.inv(C[1, 1]))

# # print(matrix_computations.generate_schur_complement_iterative(C, 63, [1]).shape)

# C_hat = matrix_computations.generate_schur_complement_iterative(C, 63, [14])
# test_mesp2 = (Mesp(C_hat))
# print(test_mesp2.n)
# print(test_mesp2.d)
# print(C[14][:, 14])
# print(C[14, 14])
# val = test_mesp2.solve_approximate(9)[0] + slogdet(C[[14]][:, [14]])[1]
# print(val)

# Check the fixing out - WORKS
# remaining_indices = np.setdiff1d(np.arange(63), [0])
# C_hat = C[remaining_indices][:, remaining_indices]
# test_mesp2 = Mesp(C_hat)
# print(test_mesp.solve_approximate(10)[0])


