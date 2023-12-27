import pandas as pd
import numpy as np
from numpy.linalg import slogdet
import os

from mesp.mesp import Mesp
from mesp.utilities.mesp_data import MespData
from mesp.utilities import matrix_computations
from mesp.branching.variable_fixing import varfix
from mesp.bounding.frankwolfe import frankwolfe

C = pd.read_table(os.getcwd() + "/examples/data/Data63.dms",
                  header=None, encoding="utf-8", sep="\s+")

n = 63
C = np.array(C)
C = C.reshape(n,n)

C = 100*np.matrix(C)

data = MespData(C)

arr = [1, 2, 3]

print(data.C[arr][:, arr].shape)

# print(data.shape)

# returned = frankwolfe(data, 20)
# # print(returned)
# print(type(returned[4]))


