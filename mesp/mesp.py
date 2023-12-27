from numpy import (matrix, ndarray, setdiff1d, where, isin, arange)
from numpy.linalg import (matrix_rank, slogdet)
from typing import Tuple, Callable, Union, List
from numbers import Number

from mesp.utilities.mesp_data import MespData
from mesp.bounding.bound_chooser import BoundChooser
from mesp.bounding.frankwolfe import frankwolfe
from mesp.utilities.matrix_computations import (generate_factorizations, generate_schur_complement, fix_out, is_psd) 
from mesp.approximation.localsearch import localsearch
from mesp.branching.variable_fixing import fix_variables
# from mesp.tree import tree
from mesp.tree.tree import Tree # TODO: What is up with this?

class Mesp:
    """
    A Maximum Entropy Sampling Problem (MESP)

    Problems are immutable in the following sense:
    - The PSD matrix defining the Mesp object cannot be changed
    - Once the solve function has been run on a Mesp object for some subset size, that Mesp object
        cannot be solved (exactly *or approximately*) for a different subset size

    However, before running the solve function, A Mesp object can be used to *approximately* solve the
    MESP for more than one subset size.

    Parameters
    ----------
    C : numpy.matrix
        The positive semidefinite matrix which defines the MESP
    """
    
    def __init__(self, C: matrix):
        
        self.C = MespData(C, factorize=True)
        self.default_chooser = BoundChooser(C, frankwolfe)

        ### ###

        self.s: int = None
        self.succ_var_fix: bool = None
        self.scale_factor: float = None # probability incorporate this into MespData object

        ### TODO: only keep one of these
        self._attempted_solve: bool = False
        self._successful_solve: bool = None
        ###

        self._approximate_value: float = None
        self._approximate_solution: List[int] = None

        self._value = None
        self._solution = None 
        self._solve_time: float = None

        self.solution_tree: Tree = None # use this to access tree/solving stats
    
    @property
    def attempted_solve(self):
        """
        bool : whether the solve function has been called yet
        """
        return self._attempted_solve
    
    @property
    def successful_solve(self):
        """
        bool : whether the solve function terminated with an exact (solution, value) pair
            (or None if the solve function hasn't been called yet)
        """
        return self._successful_solve

    @property
    def approximate_value(self):
        """
        float : the approximate value from the last time the problem was solved
            (or None if solve has not been attempted)
        """
        # if self._approximate_value == None:
        #     raise NotComputedError()
        # else:
        #     return self._approximate_value
        return self._approximate_value
    
    @property
    def approximate_solution(self):
        """
        List[int] : the approximate solution from the last time the problem was solved
            (or None if solve has not been attempted)
        """
        # if self._approximate_solution == None:
        #     raise NotComputedError()
        # else:
        #     return self._approximate_solution
        return self._approximate_solution

    @property
    def value(self):
        """
        float : the value from the last time the problem was solved
            (or None if not solved)
        """
        return self._value
    
    @property
    def solution(self):
        """
        List[int] : the solution from the last time the problem was solved
            (or None if solve has not been attempted)
        """

    """
    Other Property TODO (useful for experimentation purposes):
    - solve time
    - presolve time (varfixing)
    - number of variables fixed
    """
    
    def approximate_solve(self, s: int) -> Tuple[float, ndarray, float]:
        """
        Approximately solves the MESP

        Corresponds to algorithm developed in \"Best Principal Submatrix Selection for the Maximum Entropy Sampling Problem\"
        by Li and Xie

        Parameters
        ----------
        s : int, 0 < s <= min{rank C, n-1}
            number of measurements to select
            <=> size of the subset

        Returns
        -------
        z_hat : float
            Approximate optimal value of the MESP
        x_hat : ndarray
            Approximate solution of the MESP
        runtime : float
        """
        self.solve_checks(s)
        return localsearch(self.C.V, self.E, self.n, self.d, s)
    
    def solve(self, s: int, fix_vars: bool=True, timeout: float=60,
              bound_chooser: BoundChooser=None, tol:float=1e-3) -> Tuple[bool, float]:
        
        self.solve_checks(s)
        
        self.s = s

        if bound_chooser != None:
            if isinstance(bound_chooser, BoundChooser):
                self.default_chooser = bound_chooser
            else:
                raise TypeError("You provided an object as a BoundChooser argument which is not\
                                an actual BoundChooser.")

        if fix_vars:
            self.succ_var_fix, self.C, self.s, self.scale_factor = fix_variables(s=s, C=self.C)
        
        self._approximate_value = self.approximate_solve(self.s)[0]
        
        soln_tree = Tree(self.C, self.s, self._approximate_value, self.default_chooser,
                              scale_factor=self.scale_factor)
        
        self.solved, z_LB = soln_tree.solve_tree(timeout) # Add 

        return self.solved, z_LB
        
    
    def solve_checks(self, s: int):
        if self._attempted_solve == True:
            raise ValueError(f"You have already attempted to solve this tree\
                             with s={self.s}. You cannot attempt to solve (exactly or approximately) an MESP object\
                             more than once. An option to continue solving a problem will however\
                                be created in future iterations of this solver.")
        
        if not isinstance(s, int):
            raise ValueError(f"{s} is an improper \"s\" parameter for tree initialization. Please pass in an integer-valued \"s\".")
        
        if s == 0 or s > min(self.C.d, self.C.n-1):
            raise ValueError(f"You passed in an improper values of s.\
                             Please choose s s.t 0 < s <= min(rank C, n - 1).")
    
    
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

        ### Type checks for parameters ###
        
        # # if not isinstance(timeout, numbers.Number) or timeout < 0:
        # #     raise ValueError(f"{timeout} is an improper \"timeout\" parameter for a tree initialization. Please pass in a numeric value greater than 0.") 
        
        # if not isinstance(optimal_approx, numbers.Number):
        #     raise ValueError(f"{optimal_approx} is an improper \"optimal_approx\" parameter for a tree initialization. Please pass in a numeric value.") 
        
        # if not isinstance(branch_idx_constant, numbers.Number) or branch_idx_constant > 1 or branch_idx_constant < 0:
        #     raise ValueError(f"{branch_idx_constant} is an improper \"timeout\" parameter for a tree initialization. Please pass in a numeric value in [0, 1].")


class NotComputedError(Exception):
    """Custom exception for indicating that a computation has not been performed."""

    def __init__(self, message="Computation has not been performed."):
        self.message = message
        super().__init__(self.message)