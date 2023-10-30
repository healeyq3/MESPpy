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
V, Vsquare, E = matrix_computations.generate_factorizations(C, n, n)
# print("V Square type", type(Vsquare[0]))
S1, S0, time = varfix(V, Vsquare, E, n, n, 10)
print(S1, S0, time)
# print("V shape", V.shape)

# test_mesp = Mesp(C)
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


