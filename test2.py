import pandas as pd
import numpy as np
import os
import datetime

from mesp import Mesp

C = pd.read_table(os.getcwd() + "/examples/data/Data63.dms",
                  header=None, encoding="utf-8", sep="\s+")

n = 63
C = np.array(C)
C = C.reshape(n,n)

C = 100*np.matrix(C)

mesp1 = Mesp(C)

df_results = pd.DataFrame(columns=('s', 'Solved', 'Total Time', 'Iterations', 'GAP', 'z-hat', 'LB', 'num updates'))
loc = 0

start_test = datetime.datetime.now()

test_vals = [i for i in range(3, 11)]
# test_vals = [3]

for s in test_vals:
    print(f'\n === Testing s = {s} ===')
    solved, opt_val, time, iterations, gap, z_hat, num_updates = mesp1.solve(s)
    df_results.loc[loc] = np.array([s, solved, time, iterations, gap, z_hat, opt_val, num_updates])
    loc += 1
    df_results.to_csv('test_results1024-2.csv', index=False)
    