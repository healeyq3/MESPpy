import pandas as pd
import numpy as np
import os

# from mesp.approximation.localsearch import localsearch

import mesp 

C = pd.read_table(os.getcwd() + "/examples/data/Data63.dms",
                  header=None, encoding="utf-8", sep="\s+")

n = 63
C = np.array(C)
C = C.reshape(n,n)

C = 100*np.matrix(C)



# data1 = MespData(C)

# print(data1.shape)