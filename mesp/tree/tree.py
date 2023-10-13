from numpy import (sqrt, matrix, array, dot, diag, eye, flatnonzero)
from typing import Tuple, List
import numbers
import datetime

from utilities.matrix_computations import generate_factorizations
from node import Node

class Tree:

    def __init__(self, n: int, d: int, s: int, C: matrix, optimal_approx: float, S1: List[int] = [], S0: List[int] = [],
                 timeout: float = 0, branch_idx_constant: float = 0.5, epsilon: float = 1e-6) -> None:
        """
        Parameters
        ----------
        n : int
            Number of potential sensor placements
        d: int
            Dimension of the covariance matrix
        s :  int
            The number of measurements allowed to maximize information from the covariance matrix
        C : matrix
            The covariance matrix associated with a MESP instance
        optimal_approx : float
            The best known lower bound for the DDF problem with a given n and s
        S1: List[int], optional
            The measurements which should be selected to maximize information
        S0 : List[int], optional
            The measurements which should not be selected to maximize information
        timeout : number, optional
            Number of minutes the solve_tree function will be allowed to run before timing out. By default the timeout will not be set.
        branching_idx_constant: float, optional
            When a subproblem needs to be branched, the chosen branching index will follow argmin|x_i - branching_idx_val| such that the index has
            not already been branched and is not binary (if branching_idx_val == 0 or 1)
        epsilon : number, optional
            Numerical error parameter. Dictates the
        """

        ### Type checks for parameters ###
        if not isinstance(n, int):
            raise ValueError(f"{n} is an improper \"n\" parameter for tree initialization. Please pass in an integer-valued \"n\".")
        
        if not isinstance(s, int):
            raise ValueError(f"{s} is an improper \"s\" parameter for tree initialization. Please pass in an integer-valued \"s\".")
        
        if s >= n:
            raise ValueError('The number of new sensors (s) must be less than the number of potential placement locations. n - s:', n - s)
        
        if not isinstance(timeout, numbers.Number) or timeout < 0:
            raise ValueError(f"{timeout} is an improper \"timeout\" parameter for a tree initialization. Please pass in a numeric value greater than 0.") 
        
        if not isinstance(optimal_approx, numbers.Number):
            raise ValueError(f"{optimal_approx} is an improper \"optimal_approx\" parameter for a tree initialization. Please pass in a numeric value.") 
        
        if not isinstance(branch_idx_constant, numbers.Number) or branch_idx_constant > 1 or branch_idx_constant < 0:
            raise ValueError(f"{branch_idx_constant} is an improper \"timeout\" parameter for a tree initialization. Please pass in a numeric value in [0, 1].")

        if not isinstance(timeout, numbers.Number):
            raise ValueError(f"{timeout} is an improper \"timeout\" parameter for a tree initialization. Please pass in a numeric value.") 
        ###   ####

        ### Attribute Assignments ###
        # Data Assignments
        self.n = n
        self.d = d
        self.s = s
        self.C = C
        self.V, self.Vsquare, self.E = generate_factorizations(C, n, d)

        self.z_hat = optimal_approx
        self.z_lub = float('inf') # least upper bound
        self.z_ub = float('inf') # upper bound
        self.z_lb = optimal_approx - epsilon
        
        # Parameter assignments
        self.TIMEOUT = timeout
        self.EPS = epsilon
        self.branch_idx_val = branch_idx_constant

        # Tree properties
        self.solved = False
        self.verified = False # Whether the approximate optimal has been verified. corresponds to |z_ub - z_hat| < epsilon
        self.early_termination = False

        self.open_nodes = []
        self.processed = []
        self.updated_lb_iterations = [] # iterations which corresponded to z_lb being updated
        self.optimal_node = None 

        # Node Class Attributes 
        Node.branch_idx_constant = branch_idx_constant
        Node.C = C
        Node.n = n
        Node.d = d
        Node.s = s 

        # Begin Queue
        root = Node(0, 1, 1, C, S0, S1, self.V, self.Vsquare, self.E)
        self.open_nodes.append(root)
        self.delta_criterion = None

        ### ###

        ### Framework Statistics ###
        self.z_inital_bound = None # Corresponding to bound found for root subproblem
        self.gap = None # The gap between z_inital_bound and z_hat
        self.node_counter = 1
        self.log = ""
        self.total_time = 0 # total time it takes to solve the tree
        self.total_solve_time = 0 # total time it takes to compute the bounds
        self.total_iterations = 0
        self.num_solved = 0 # the number of times a subproblem's bound had to be computed
    
    ### End of Constructor ###

    ### Begin methods used for solving tree ###

    def solve_tree(self):

        solve_time_start = datetime.datetime.now()

        while self.open_nodes:

            iteration_start = datetime.datetime.now()

            node = self.open_nodes.pop()

            if not node.is_solved:
                solve_time = self.solve_node(node=node)
                self.total_solve_time += solve_time
            
            self.evaluate_node(node, solve_time_start)

    def solve_node(self, node: Node) -> float:
        solve_time = node.compute_subproblem_bound
        if node.id == 1:
            self.z_max = node.relaxed_z
            self.gap = self.z_max - self.z_hat
            self.delta_criterion = abs(self.gap) / 2
        
        self.num_solved += 1

        return solve_time
    
    def evaluate_node(self, node: Node, runtime_start):
        
        z = node.relaxed_z
        is_integral = node.is_integral

        if z >= self.z_lb and is_integral:
            self.z_lb = z
            self.optimal_node = node
            self.updated_lb_iterations.append(self.total_iterations + 1)
        
        elif z > self.z_lb and not is_integral:
            curr_time = (datetime.datetime.now() - runtime_start).total_seconds()

            if self.TIMEOUT == 0 or self.TIMEOUT * 60 > curr_time:
                # DEBUG - compute branching index
                node.compute_branch_index()
                left_node, right_node, right_branch = self.branch(node)
                self.add_nodes(left_node, right_node, right_branch)
            else:
                self.early_termination = True
                # DEBUG - CHECK BOUNDS
        
        self.processed.append(node)

    def branch(self, node: Node):
        pass
        # right_branch = None

        # if node

