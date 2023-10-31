from numpy import (matrix, ndarray, setdiff1d, where, isin, arange)
from numpy.linalg import (matrix_rank, slogdet)
from typing import Tuple, List

from mesp.utilities.matrix_computations import (generate_factorizations, generate_schur_complement, fix_out) 
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

        self.s = None

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
    
    def fix_variables(self, s: int) -> Tuple[bool, matrix, int, int, int, float]:
        S1, S0, _ = varfix(self.V, self.Vsquare, self.E, self.n, self.d, s)
        if S1 == 0 and S0 == 0:
            print("No variables could be fixed")
            return False, None, None, None, None, None
        else:
            C_hat = self.C
            n_hat = self.n
            d_hat = self.d
            s_hat = s
            scale_factor = 0
            if len(S0) > 0:
                C_hat = fix_out(self.C, S0)
                n_hat = n_hat - len(S0)
                d_hat = d_hat - len(S0)
            if len(S1) > 0:
                remaining_indices = setdiff1d(arange(self.n), S0)
                updated_indices = where(isin(remaining_indices, S1))[0]
                #### Scaling ###
                C_ff = C_hat[updated_indices][:, updated_indices]
                scale_factor += slogdet(C_ff)[1]
                ### ###
                C_hat = generate_schur_complement(C_hat, n_hat, updated_indices)
                n_hat = n_hat - len(S1)
                s_hat = s_hat - len(S1)
                d_hat = d_hat - len(S1)
            return True, C_hat, n_hat, d_hat, s_hat, scale_factor
    
    def solve(self, s: int, fix_vars: bool = True, timeout: float=60):
        # Add check to see if already attempted to solve.
        self.s = s
        scale_factor = 0
        if fix_vars:
            succ_fix, C_hat, n_hat, d_hat, s_hat, scale = self.fix_variables(s)
            if succ_fix:
                self.n = n_hat
                self.d = d_hat
                self.s = s_hat
                self.C = C_hat
                self.V, self.Vsquare, self.E = generate_factorizations(self.C, self.n, self.d)
                scale_factor += scale
        z_hat = self.solve_approximate(self.s)[0]
        milp = tree.Tree(self.n, self.d, self.s, self.C, z_hat, scale_factor=scale_factor)
        solved, opt_val, time, iterations, gap, num_updates = milp.solve_tree()
        return solved, opt_val, time, iterations, gap, z_hat, num_updates

