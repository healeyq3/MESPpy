import pandas as pd
import numpy as np
import os

from mesp import Mesp

C = pd.read_table(os.getcwd() + "/examples/data/Data63.dms",
                  header=None, encoding="utf-8", sep="\s+")

n = 63
C = np.array(C)
C = C.reshape(n,n)

C = 100*np.matrix(C)

test_mesp = Mesp(C)
# print(test_mesp.obj_f([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]))
print(test_mesp.solve_approximate(10))