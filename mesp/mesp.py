from numpy import (matrix, ndarray)
from typing import Tuple, Callable, Union, List
from numbers import Number
import time

from mesp.utilities.mesp_data import MespData
from mesp.bounding.bound_chooser import BoundChooser
from mesp.bounding.frankwolfe import frankwolfe
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

    Raises
    ------
    ValueError if the provided C is not PSD
    """
    
    def __init__(self, C: matrix):
        
        init_time = time.time()
        self.C = MespData(C, factorize=True)
        init_time = time.time() - init_time
        
        self.default_chooser = BoundChooser(C, frankwolfe)

        ### ###

        self._s: int = None
        self._tol: float = 1e-6

        self._succ_var_fix: bool = None
        self._fixed_in: List[int] = []
        self._fixed_out: List[int] = []
        self._var_fix_time: float = None

        self._successful_solve: bool = None
        self._verified_solution: bool = None

        self._approximate_value: float = None
        self._approximate_solution: List[int] = None

        self._value = None
        self._solution = None 
        self._compilation_time = init_time
        self._solve_time: float = None

        self._solution_tree: Tree = None # use this to access tree/solving stats - include time_to_opt
    
    @property
    def s(self):
        """
        int : size of the subset the MESP is being solved for
            (or None if the solve function hasn't been called yet)
        """
        return self._s
    
    @property
    def tol(self):
        """
        float : the tolerance to which one wants to verify the exact solution
        """
        return self._tol
    
    @property
    def succ_var_fix(self):
        """
        bool : whether there are variables in this MESP which can be fixed in our out
            (or None if the solve function hasn't been called yet)
        int : the number of successfully fixed in variables 
        int : the number of successfully fixed out variables
        """
        return self._succ_var_fix, len(self._fixed_in), len(self._fixed_out)
    
    @property
    def fixed_in(self):
        """
        List[int] : variables fixed into the problem <=> variables known to be == 1

        Raises
        ------
        NotComputedError if variable fix hasn't been computed <=> solve hasn't been called
        """
        if self._succ_var_fix == None:
            raise NotComputedError()
        else:
            return self._fixed_in
        
    @property
    def fixed_out(self):
        """
        List[int] : variables fixed out of the problem <=> variables known to be == 0

        Raises
        ------
        NotComputedError if variable fix hasn't been computed <=> solve hasn't been called
        """
        if self._succ_var_fix == None:
            raise NotComputedError()
        else:
            return self._fixed_out
    
    @property
    def var_fix_time(self):
        """
        float : time it took to find the fixed variables

        Raises
        ------
        NotComputedError if variable fix hasn't been computed <=> solve hasn't been called
        """
        if self._succ_var_fix == None:
            raise NotComputedError()
        else:
            return self._var_fix_time
    
    @property 
    def compilation_time(self):
        """
        float: time (in seconds) it took to compile the problem. This includes the time it took
            to create the MESP data object, perform variable fixing, compute the local solution, 
            and initialize the tree. (Note that depending on when accesses this property
            it is possible only some of these actions have been performed)
        """
        return self._compilation_time
    
    @property
    def attempted_solve(self):
        """
        bool : whether the solve function has been called yet
        """
        if self._successful_solve == None:
            return False
        else:
            return True
    
    @property
    def successful_solve(self):
        """
        bool : whether the solve function terminated with a newly fathomed (solution, value) pair.
            (or None if the solve function hasn't been called yet).
        """
        return self._successful_solve

    @property
    def verified_solution(self):
        """
        bool : whether the solve function terminated with a solution whose value
            is within some tolerance of the optimal solution. This value could be
            the value provided by the approximate algorithm, or it could have been
            generated by the solve function. Note that successful solve => verfied_solution,
            but the converse is not true.
            (or None if the solve function hasn't been called yet).
        """
        return self._verified_solution
    
    @property
    def approximate_value(self):
        """
        float : the approximate value from the last time the problem was solved
            (or None if solve has not been attempted)
        """
        return self._approximate_value
    
    @property
    def approximate_solution(self):
        """
        List[int] : the approximate solution from the last time the problem was solved
            (or None if solve has not been attempted)
        """
        return self._approximate_solution

    @property
    def value(self):
        """
        float : the value from the last time the problem was solved.
            This value could either be the optimal value (to some tolerance),
            or the value of a good lower bound. See the solution properties to 
            determine which it is.
            (or None if not solved)
        """
        return self._value
    
    @property
    def solution(self):
        """
        np.ndarray[int] or List[int] : the solution from the last time the problem was solved.
            This solution could either be the optimal solution (found to some tolerance),
            or the value of a good lower bound. See the solution properties to determine 
            which it is.
            (or None if solve has not been attempted)
        """
        return self._solution
    
    @property
    def upper_bound(self):
        """
        float : if the solve function terminates without having verified an optimal solution,
            this upper_bound property is the largest continuous relaxation of any node in the 
            non-fully enumerated tree (TODO: since early termination is not yet allowed).
            Note that if the tree is verified, then this value is approximately the optimal value.

        Raises
        ------
        NotComputedError : if the solve function has not yet been called.
        """
        if self._successful_solve == None:
            raise NotComputedError()
        else:
            return self._solution_tree.z_ub
    
    @property
    def approximate_verified(self):
        """
        bool : Whether the solution found by the approximation algorithm is the optimal solution (given some tol)

        Raises
        ------
        NotComputedError : if the solve function has not yet been called.
        """
        if self._successful_solve == None:
            raise NotComputedError()
        else:
            return self._solution_tree.approximate_verified
    
    @property
    def time_to_opt(self):
        """
        float : the number of seconds it took to verify an optimal solution.
            Note that it may take much longer to fathom an optimal solution

        Raises
        ------
        NotComputedError : if the solve function has not yet been called.
        """
        if self._successful_solve == None:
            raise NotComputedError()
        else:
            return self._solution_tree.time_to_opt
    
    @property
    def solve_time(self):
        """
        float : the number of seconds it took to solve the problem
            (or None if not solved)
        """
        return self._solve_time
    
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

        Raises
        ------
        AlreadyComputedError
            Raised if the solve function has already been called for this MESP object
        ValueError
            Raises if the provided s value is either not an integer or does not meet
            MESP formulation requirements.
        """
        self._solve_checks(s)
        return localsearch(self.C.V, self.C.E, self.C.n, self.C.d, s)
    
    def solve(self, s: int, fix_vars: bool=True, timeout: float=60,
              bound_chooser: BoundChooser=None, tol:float=1e-6) -> Tuple[bool, float]:
        """
        Attempts to exactly solves the MESP.

        The solve function constructs and then enumerates a branch and bound tree where the continuous relaxation
        of nodes (equivalently, subproblems) in the tree is provided (by default) by the bound found in
        \"Best Principal Submatrix Selection for the Maximum Entropy Sampling Problem,\" which is solved using an 
        efficient Frank-Wolfe algorithm.

        Furthermore, as it is currently implemented, the solve function will always attempt to fathom a solution to the MESP,
        even if the largest continuous relaxation in the queue of nodes (referred to as "the tree's upper bound") is within
        a user-specified tolerance range to the tree's lower bound (the largest optimal value associated with a feasible solution;
        originally provided as z_hat). 
        While this "time_to_opt" can be accessed later, the tree will continue to be enumerated even after this optimality verification 
        has been reached.

        Upon termination, the algorithm will specify whether a new solution was fathomed, whether an optimal solution was verified,
        the value associated with an optimal solution (which can be z_hat if the tree failed to fathom a new solution), and the runtime of
        the attempted solve.
        
        TODO: Implement early termination and implement "warm start."

        Parameters
        ----------
        s : int, 0 < s <= min{rank C, n-1}
            Number of measurements to select
            <=> size of the principal subset defining the MESP
        fix_vars : bool, optional
            When True, before the B&B tree is compiled, the dual variables
            of the continuous relaxation of the original problem will be used to (potentially) fix
            decision variables to 1 or 0. The original problem will then be shrunk according
            to fixing these variables in or out before being . Defaults to True.
        timeout : float, optional
            How many minutes the B&B tree will be allowed to add new nodes to its open queue. 
            After the timeout, the nodes in the queue will be evaluated without 
            Pass in 0 if you do not wish for there to be a timeout.
        bound_chooser: BoundChooser, optional
            How the upper bounds are found for each subproblem in the tree
        tol : float, optional
            How close (as measured by Hamming distance) the tree's upper bound should be to 
            the approximate solution OR the lower bound to guarantee solution optimality.

        Returns
        -------
        bool : 
            Whether the tree *generated* a new solution
        bool : 
            Whether the tree guarantees an optimal solution
        float
            max{z_LB, z_hat}
        ndarray :
            solution associated with above float
        float
            runtime (in seconds)

        Raises
        ------
        AlreadyComputedError
            Raised if the solve function has already been called for this MESP object
        ValueError
            Raised if the provided s value is either not an integer or does not meet
            MESP formulation requirements.
        TypeError
            Raised if a provided bound_chooser is not actually of class BoundChooser

        """
        
        self._solve_checks(s)

        self._param_checks(timeout, bound_chooser, tol)
        
        self._s = s

        if bound_chooser != None:
            self.default_chooser = bound_chooser

        if fix_vars:
            self._succ_var_fix, self.C, self._s, self._var_fix_time = fix_variables(s=s, C=self.C)
            self._compilation_time += self._var_fix_time
            self._fixed_in = self.C.S1
            self._fixed_out = self.C.S0
        
        self._approximate_value, _, approximate_solve_time = self.approximate_solve(self.s)
        self._compilation_time += approximate_solve_time
        
        tree_comp_time_start = time.time()
        self._solution_tree: Tree = Tree(self.C, self.s, self._approximate_value, self.default_chooser)
        self._compilation_time += time.time() - tree_comp_time_start
        
        self._successful_solve, self._verified_solution, self._value, self._solution, self._solve_time = self._solution_tree.solve_tree(tol=tol, timeout=0)

        return self._successful_solve, self._verified_solution, self._value, self._solve_time
        
    
    def _solve_checks(self, s: int):
        if self._successful_solve != None:
            raise AlreadyComputedError(f"You have already attempted to solve this tree\
                             with s={self.s}. You cannot attempt to exactly or approximately solve an MESP object\
                             more than once. An option to continue solving a problem will however\
                                be created in future iterations of this solver.")
        
        if not isinstance(s, int):
            raise ValueError(f"{s} is an improper \"s\" parameter for tree initialization. Please pass in an integer-valued \"s\".")
        
        if s == 0 or s > min(self.C.d, self.C.n-1):
            raise ValueError("You passed in an improper values of s.\
                             Please choose s subject to 0 < s <= min(rank C, n - 1).")
        
    def _param_checks(self, timeout: float, bound_chooser: BoundChooser, tol: float):
        if not isinstance(timeout, Number) or timeout < 0:
            raise ValueError(f"{timeout} is an improper \"timeout\" parameter for a tree initialization.\
                             Please pass in a numeric value greater than 0.")
        
        if bound_chooser != None and not isinstance(bound_chooser, BoundChooser):
                raise TypeError("You provided an object as a BoundChooser argument which is not\
                                an actual BoundChooser.")
        
        if not isinstance(tol, Number) or tol < 0:
            raise ValueError(f"{tol} is an improper \"tol\" parameter for a tree initialization.\
                             Please pass in a numeric value greater than 0.")
        
        if tol > 1e-1:
            print("NOTE TO USER: Your chosen tolerance is very large!\
                  Be sure that this is what you want!")



class NotComputedError(Exception):
    """Custom exception for indicating that a computation has not been performed."""

    def __init__(self, message="Computation has not been performed."):
        self.message = message
        super().__init__(self.message)

class AlreadyComputedError(Exception):
    """Custom exception for indicating that a computation has ALREADY been performed."""

    def __init__(self, message="Computation has already been performed. Recalls are not yet supported."):
        self.message=message
        super().__init__(self.message)