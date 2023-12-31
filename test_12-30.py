import pandas as pd
import numpy as np
import os
import datetime

from mesp import Mesp
from mesp.bounding.frankwolfe import frankwolfe

C = pd.read_table(os.getcwd() + "/examples/data/Data63.dms",
                  header=None, encoding="utf-8", sep="\s+")

n = 63
C = np.array(C)
C = C.reshape(n,n)

C = 100*np.matrix(C) # scale factor necessary for C \in S^n where n < 100

mesp1 = Mesp(C)

z_hat, _, _= mesp1.approximate_solve(s=5)
print("z_hat = ", z_hat)

root_UB, root_x, _, _, _ = frankwolfe(mesp1.C, s=5)
print("frank wolfe bound on root has value: ", root_UB)

print("GAP: ", abs(root_UB - z_hat))

succ_solve, verified_solution, value, solve_time = mesp1.solve(s=5, tol=1e-3)
print()
print("succ_solve (T/F): ", succ_solve)
print("verified_solution (T/F): ", verified_solution)
print("value of optimal solution: ", value)
print("tree enumeration time: ", solve_time)

print("time to opt: ", mesp1.time_to_opt)