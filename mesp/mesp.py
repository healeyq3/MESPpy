from math import floor
from numpy import (matrix, ndarray, setdiff1d, where, isin, arange)
from numpy.linalg import (matrix_rank, slogdet)
from typing import Tuple, Callable, Union, List
from numbers import Number

from mesp.utilities.mesp_data import MespData
from mesp.utilities.matrix_computations import (generate_factorizations, generate_schur_complement, fix_out, is_psd) 
from mesp.approximation.localsearch import localsearch
from mesp.branching.variable_fixing import varfix
from mesp.tree import tree

class Mesp:
    """
    Assumes an instance of the problem is solved for only one s 
    <=> cannot call solve on the same Mesp object with multiple s values
    """
    
    def __init__(self, C: matrix):
        
        self.C = MespData(C, generate_factorizations)

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
        Solves the MESP approximately

        Corresponds to 

        Parameters
        ----------
        s : number of measurements to select

        Returns
        -------
        z_hat : float
        x_hat : ndarray
        runtime : float
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
    
    # def __default_bound(self) -> Callable[[ndarray, int, int, int], ]
    
    # def solve(self, s: int, fix_vars: bool = True, timeout: float=60,
    #           choose_bound: BoundChooser=None):
        
    #     # Add check to see if already attempted to solve.
    #     self.s = s
    #     scale_factor = 0
    #     if fix_vars:
    #         succ_fix, C_hat, n_hat, d_hat, s_hat, scale = self.fix_variables(s)
    #         if succ_fix:
    #             self.n = n_hat
    #             self.d = d_hat
    #             self.s = s_hat
    #             self.C = C_hat
    #             self.V, self.Vsquare, self.E = generate_factorizations(self.C, self.n, self.d)
    #             scale_factor += scale
    #     z_hat = self.solve_approximate(self.s)[0]
    #     milp = tree.Tree(self.n, self.d, self.s, self.C, z_hat, scale_factor=scale_factor)
    #     solved, opt_val, time, iterations, gap, num_updates = milp.solve_tree()
    #     return solved, opt_val, time, iterations, gap, z_hat, num_updates


class BoundChooser:

    def __init__(self, C: matrix, default_algo: Callable[[ndarray]]) -> None:
        self.C = MespData(C)
        self.rule_checker(default_algo)
        self.default_algo = default_algo
        self.algorithm_dict = None

    def rule_checker(self, bounding_algo):
        """
        Ensures the provided bounding algorithm can accept a MespData object and an integer
        and that it returns either a tuple of the bound value, associated relaxed solution,
        and runtime, or a tuple with those three ojects and the two dual arrays associated 
        with the relaxed approximation.
        """
        s = arange(self.n)[floor(self.n / 2)]
        try:
            returned = bounding_algo(self.C, s)
            if len(returned) != 3 or len(returned) != 5:
                raise ValueError("The provided bounding algorithm didn't return the required number of arguments")
            elif len(returned) == 3 or len(returned) == 5:
                if not isinstance(returned[0], Number):
                    raise ValueError("The first returned argument (the bound) is not a Number.")
                if not isinstance(returned[1], List[float]):
                    raise ValueError("The second returned argument (x) is not a list of floats.")
                if not isinstance(returned[2], Number):
                    raise ValueError("The third returned argument (runtime) is not a Number.")
            if len(returned) == 5:
                if not isinstance(returned[3], List[float]):
                    raise ValueError("The fourth returned argument (w) is not a list of floats.")
                if not isinstance(returned[5], List[float]):
                    raise ValueError("The fifth returned argument (v) is not a list of floats.")
        except:
            raise ValueError("The provided bounding algorithm could not accept the required arguments")

    def set_bound(self, s_int: Tuple[int, int], n_int: Tuple[int, int],
                  bound_algo: Callable[
                      ...,
                       Union[Tuple[float, List[float], float],
                            Tuple[float, List[float], float, List[float], List[float]]
                       ]]) -> None:
        
        # First check if passed in bounding function meets requirements
        self.rule_checker(bound_algo)
        
        if self.algorithm_dict == None:
            n_range = arange(3, self.C.n + 1)
            s_range = arange(3, self.C.n - 3)
            
            new_bound_s = arange(s_int[0], s_int[1] + 1)
            new_bound_n = arange(n_int[0], n_int[1] + 1)
            
            self.algorithm_dict = {n : {s : bound_algo if s in new_bound_s and n in new_bound_n else self.default_algo for s in s_range} for n in n_range}

    def get_bound(self, s: int, n: int):
        if self.algorithm_dict == None:
            return self.default_algo
        else:
            try:
                algo = self.algorithm_dict[s][n]
                return algo
            except:
                return self.default_algo
