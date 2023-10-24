from numpy import matrix
from numpy.linalg import matrix_rank

from mesp.utilities.matrix_computations import (generate_factorizations, obj_f) 
from mesp.approximation.localsearch import localsearch
from mesp.tree import tree

class Mesp:
    
    def __init__(self, C: matrix):

        self.C = C

        ## size of the problem
        self.n = C.shape[0]
        self.d = matrix_rank(self.C)

        self.V, self.Vsquare, self.E = generate_factorizations(self.C, self.n, self.d)

    def return_objective_value(self):
        pass
        # return obj_f()
    
    def solve_approximate(self, s):
        return localsearch(self.V, self.E, self.n, self.d, s)
    
    def solve(self, s):
        z_hat = self.solve_approximate(s)[0]
        milp = tree.Tree(self.n, self.d, s, self.C, z_hat, timeout=60)
        solved, opt_val, time, iterations, gap, num_updates = milp.solve_tree()
        return solved, opt_val, time, iterations, gap, z_hat, num_updates
        # if solved:
        #     return solved, opt_val, time, iterations, gap
        # else:
        #     return False, False, False, False, False

