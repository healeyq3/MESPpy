from numpy import (matrix, ndarray, setdiff1d, where, isin, arange)
from numpy.linalg import (matrix_rank, slogdet)
from typing import Tuple, List

from mesp.utilities.matrix_computations import (generate_factorizations, obj_f, generate_schur_complement, fix_out) 
from mesp.approximation.localsearch import localsearch
from mesp.branching.variable_fixing import varfix
from mesp.tree import tree

class Mesp:
    """
    Assumes an instance of the problem is solved for only one s 
    <=> cannot call solve on the same Mesp object with multiple s values
    """
    
    def __init__(self, C: matrix):

        ### Problem Data ###
        
        self.C = C

        ## size of the problem
        self.n = C.shape[0]
        self.d = matrix_rank(self.C)

        self.V, self.Vsquare, self.E = generate_factorizations(self.C, self.n, self.d)

        self.s == None

        ### ###

        ### Problem Attributes ###
        # self.variables_fixed = False
        # self.successful_fix = False
        # self.S0, self.S1 = None, None

        self.approximate_solution = None
        self.approximate_value = None

        self.solved = False
        self.successful_solve = False
    
    def solve_approximate(self, s) -> Tuple[int, ndarray, int]:
        """
        Returns approximate value, approximate solution, algorithm run time
        """
        return localsearch(self.V, self.E, self.n, self.d, s)
    
    def fix_variables(self, s: int) -> Tuple[bool, matrix, float]:
        S1, S0, _ = varfix(self.V, self.Vsquare, self.E, self.n, self.d, s)
        if S1 == 0 and S0 == 0:
            print("No variables could be fixed")
            return False, None, None
        else:
            C_hat = self.C
            n_hat = self.n
            s_hat = self.s
            if len(self.S0) > 0:
                C_hat = fix_out(self.C, S0)
                n_hat = n_hat - len(self.S0)
            if len(self.S1) > 0:
                remaining_indices = setdiff1d(arange(self.n), S0)
                updated_indices = where(isin(remaining_indices, S1))
                #### Scaling ###
                C_ff = C_hat[updated_indices][:, updated_indices]
                scale_factor = slogdet(C_ff)[1]
                ### ###
                C_hat = generate_schur_complement(C_hat, n_hat, updated_indices)
                n_hat = n_hat - len(S1)
                s_hat = s_hat - len(S1)
            return True, C_hat, scale_factor
    
    def solve(self, s: int, fix_vars: bool = True, timeout: float=60):
        solve_C = self.C
        if fix_vars:
            fixed, C_hat = self.fix_variables(s)
            if fixed:
                solve_C = C_hat
        z_hat = self.solve_approximate(s)[0]
        milp = tree.Tree(self.n, self.d, s, self.C, z_hat, timeout=timeout)
        solved, opt_val, time, iterations, gap, num_updates = milp.solve_tree()
        return solved, opt_val, time, iterations, gap, z_hat, num_updates

