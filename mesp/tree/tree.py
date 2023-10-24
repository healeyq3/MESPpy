from numpy import (sqrt, matrix, zeros, ones, arange)
from typing import Tuple, List
import numbers
import datetime

from mesp.utilities.matrix_computations import (generate_factorizations, obj_f)
from mesp.tree.node import IterativeNode

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
        # Data Assignments # DEBUG - put only in one place?
        self.n = n
        self.d = d
        self.s = s
        self.C = C
        V, Vsquare, E = generate_factorizations(C, n, d) 

        self.z_hat = optimal_approx
        self.z_lub = float('inf') # least upper bound
        self.z_ub = float('inf') # upper bound
        self.z_lb = optimal_approx - epsilon
        
        # Parameter assignments
        self.TIMEOUT = timeout
        self.EPS = epsilon

        # Tree properties
        self.solved = False
        self.verified = False # Whether the approximate optimal has been verified. corresponds to |z_ub - z_hat| < epsilon
        self.early_termination = False

        self.open_nodes = []
        self.updated_lb_iterations = [] # iterations which corresponded to z_lb being updated
        self.optimal_node = None 

        # Begin Queue and Assign Node Class attributes
        IterativeNode.branch_idx_constant = branch_idx_constant

        root = IterativeNode(0, 1, 1, C, V, Vsquare, E, s, 0, False) # note the branch index and fixed_in params are ignored for root
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

            self.total_iterations += 1
            iteration_length = (datetime.datetime.now() - iteration_start).total_seconds()
            self.total_time += iteration_length

            print(f"Iteration {self.total_iterations} | current LB = {self.z_lb} | Number of Open Subproblems = {len(self.open_nodes)}"
                + f" | Total Running Time = {self.total_time} seconds ", end = "\r") 
            
        self.total_time = datetime.datetime.now() - solve_time_start
        return self.evaluate_tree()


    def solve_node(self, node: IterativeNode) -> float:
        solve_time = node.compute_subproblem_bound()
        if node.id == 1:
            self.z_inital_bound = node.relaxed_z
            self.gap = self.z_inital_bound - self.z_hat
            self.delta_criterion = abs(self.gap) / 2
        
        self.num_solved += 1

        return solve_time
    
    def evaluate_node(self, node: IterativeNode, runtime_start):
        
        z = node.relaxed_z
        is_integral = node.is_integral

        if z >= self.z_lb and is_integral:
            self.z_lb = z
            self.optimal_node = node
            self.updated_lb_iterations.append(self.total_iterations + 1)
        
        elif z > self.z_lb and not is_integral:
            curr_time = (datetime.datetime.now() - runtime_start).total_seconds()

            if self.TIMEOUT == 0 or self.TIMEOUT * 60 > curr_time:
                # node.compute_branch_index()
                left_node, right_node, right_branch = self.branch(node)
                self.add_nodes(left_node, right_node, right_branch)
            else:
                self.early_termination = True
                # DEBUG - CHECK BOUNDS

    def branch(self, node: IterativeNode) -> Tuple[IterativeNode, IterativeNode, bool] :
        
        right_branch = None

        # if node.delta_i_max > self.delta_criterion: # use dual branching strategy
        #     branch_idx = node.i_max
        #     if node.w_branch == True:
        #         right_branch = True
        #     else:
        #         right_branch = False
        # else:
        #     branch_idx = node.backup_branch_idx

        branch_idx = node.backup_branch_idx
        left_node = None
        right_node = None

        if (node.s_curr == node.n_curr - 1):
            self.enumerate_S0(node)
        elif (node.s_curr == 1):
            self.enumerate_S1(node)
        else:
            self.node_counter += 1
            left_node = IterativeNode(node.id, self.node_counter, node.depth + 1, node.C_hat, node.V_hat, node.Vsquare_hat,
                                      node.E_hat, node.s_curr, branch_idx, fixed_in=False, scale_factor=node.scale_factor)
            self.node_counter += 1
            right_node = IterativeNode(node.id, self.node_counter, node.depth + 1, node.C_hat, node.V_hat, node.Vsquare_hat,
                                      node.E_hat, node.s_curr, branch_idx, fixed_in=True, scale_factor=node.scale_factor)

        return left_node, right_node, right_branch
    
    def enumerate_S0(self, node: IterativeNode):
        for i in arange(node.n_curr):
            x = ones(node.n_curr)
            x[i] = 0
            z = obj_f(x, node.Vsquare_hat) + node.scale_factor
            if z > self.z_lb:
                # self.node_counter += 1 
                self.z_lb = z
                # self.optimal_node DEBUG - Need a set solution method
                self.updated_lb_iterations.append(self.total_iterations + 1)


    def enumerate_S1(self, node: IterativeNode):
        for i in arange(node.n_curr):
            x = zeros(node.n_curr)
            x[i] = 1
            z = obj_f(x, node.Vsquare_hat) + node.scale_factor
            if z > self.z_lb:
                self.z_lb = z
                self.updated_lb_iterations.append(self.total_iterations + 1)

    def add_nodes(self, left_node, right_node, right_branch) -> None:
        if right_branch == None:
            if self.s <= 40:
                self.right_node_first(left_node, right_node)
            else:
                self.left_node_first(left_node, right_node)
        elif right_branch:
            if self.node_counter <= 3: # so for the root node
                self.right_node_first(left_node, right_node)
            else:
                self.left_node_first(left_node, right_node)
        else:
            if self.node_counter <= 3:
                self.left_node_first(left_node, right_node)
            else:
                self.right_node_first(left_node, right_node)
    
    def right_node_first(self, left_node, right_node):
        """
        Adds the right_node to the queue first (will be processed second)
        """
        if right_node != None:
            self.open_nodes.append(right_node)
        if left_node != None:
            self.open_nodes.append(left_node)

    
    def left_node_first(self, left_node, right_node):
        if left_node != None:
            self.open_nodes.append(left_node)
        if right_node != None:
            self.open_nodes.append(right_node)

    def evaluate_tree(self):
        # if not self.early_termination:
        #     return self.solved, self.z_lb, self.total_time, self.total_iterations, self.gap
        # else:
        #     return False
        num_updates = len(self.updated_lb_iterations)
        if num_updates > 0:
            solved = True
        else:
            solved = False 
        return solved, self.z_lb, self.total_time, self.total_iterations, self.gap, num_updates