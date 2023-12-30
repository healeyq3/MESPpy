from numpy import (sqrt, matrix, zeros, ones, arange)
from typing import Tuple, List
import numbers
import datetime

from mesp.utilities.mesp_data import MespData
from mesp.utilities.matrix_computations import (generate_factorizations, obj_f)
from mesp.tree.node import IterativeNode
from mesp.bounding.bound_chooser import BoundChooser

class Tree:

    def __init__(self, C: MespData, s: int, optimal_approx: float, bound_chooser: BoundChooser,
                 branch_idx_constant: float = 0.5, epsilon: float = 1e-6) -> None:
        """
        Parameters
        ---------- 
        C : MespData
            The covariance matrix associated with a MESP instance
        s : int
            The number of measurements allowed to be selected from the candidate set to 
            maximize the information.
        optimal_approx : float
            The best known lower bound for the DDF problem with a given n and s
        branching_idx_constant: float, optional
            When a subproblem needs to be branched, the chosen branching index will follow argmin|x_i - branching_idx_val| such that the index has
            not already been branched and is not binary (if branching_idx_val == 0 or 1)
        epsilon : number, optional
            Numerical error parameter. Dictates the drop of z_hat with the goal of requiring
            the tree to generate an optimal solution
        """
        
        # Parameter assignments
        self.TIMEOUT = 0
        self.EPS = epsilon

        # Assign Node Class attributes
        IterativeNode.branch_idx_constant = branch_idx_constant
        IterativeNode.bound_chooser = bound_chooser
    
        # note the branch index and fixed_in params are ignored for root
        # Also note that the time it takes to bound the roote node is not 
        # added to the tree's solution time, but instead is tracked in the
        # MESP compilation time
        root : IterativeNode = IterativeNode(0, 1, 1, C, s, 0, False) 
        self.open_nodes = [root]

        self.z_hat = optimal_approx + C.scale_factor
        self.z_lb = optimal_approx + C.scale_factor - epsilon

        self.z_inital_bound = root.relaxed_z
        self.z_lub = self.z_inital_bound
        self.z_ub = self.z_inital_bound
        self.ub_index = 1 # TODO: will I be using this for global UB branching strategy?
        self.gap = self.z_inital_bound - self.z_hat
        self.delta_criterion = abs(self.gap) / 2

        ### Framework Statistics ###
        self.node_counter = 1
        self.log = ""
        self.total_time = root.solve_time # total time it takes to solve the tree
        self.solve_times = [root.solve_time]
        self.total_iterations = 1 
        self.num_solved = 1 # the number of times a subproblem's bound had to be computed
        self.updated_lb_iterations = [] # iterations which corresponded to z_lb being updated
        self.dual_branched = [] # iterations which branched using the dual variables
        self.num_updates = 0
        self.optimal_node = None 

        self.solved = False
        self.verified = False # Whether the approximate optimal has been verified. corresponds to |z_ub - z_hat| < tol (parameter passed in solve function)
        self.early_termination = False
    
    ### End of Constructor ###

    ### Begin methods used for solving tree ###

    def solve_tree(self, timeout: float, tol: float):

        if timeout != None:
            self.TIMEOUT = timeout

        solve_time_start = datetime.datetime.now()

        while self.open_nodes and self.z_ub > self.z_lb:
            # linear search through open nodes for global max at each iteration
            # could cause exit of loop due to second condition <=> list could be nonempty,
            # but optimal solution found
            # QUESTION: incorproating tolerance

            iteration_start = datetime.datetime.now()

            node: IterativeNode = self.open_nodes.pop()

            self.solve_time.append(node.solve_time)
            
            self.evaluate_node(node, solve_time_start)

            self.total_iterations += 1
            iteration_length = (datetime.datetime.now() - iteration_start).total_seconds()
            self.total_time += iteration_length

            print(f"Iteration {self.total_iterations} | current LB = {self.z_lb} | Number of Open Subproblems = {len(self.open_nodes)}"
                + f" | Total Running Time = {self.total_time} seconds ", end = "\r") 
            
        self.total_time = datetime.datetime.now() - solve_time_start
        return self.evaluate_tree()
           
    def evaluate_node(self, node: IterativeNode, runtime_start):
        
        # Will need to make parent node class 

        z = node.relaxed_z
        is_integral = node.integral

        if z >= self.z_lb and is_integral:
            self.z_lb = z
            self.optimal_node = node
            self.updated_lb_iterations.append(self.total_iterations + 1)
            self.num_updates += 1
            # ADD UPPER BOUND CHECK METHOD CALL
        
        elif z > self.z_lb and not is_integral:
            curr_time = (datetime.datetime.now() - runtime_start).total_seconds()

            if self.TIMEOUT == 0 or self.TIMEOUT * 60 > curr_time:
                node.compute_branch_index()
                left_node, right_node, right_branch = self.branch(node)
                self.add_nodes(left_node, right_node, right_branch)
            else:
                self.early_termination = True
                # DEBUG - CHECK BOUNDS

    def branch(self, node: IterativeNode) -> Tuple[IterativeNode, IterativeNode, bool] :

        right_branch = None
        if node.delta_i_max > self.delta_criterion: # use dual branching strategy
            self.dual_branched.append(self.total_iterations)
            branch_idx = node.i_max
            if node.w_branch == True:
                right_branch = True
            else:
                right_branch = False
        else:
            branch_idx = node.backup_branch_idx

        left_node = None
        right_node = None

        if (node.s_curr == node.n_curr - 1):
            self.enumerate_S0(node)
        elif (node.s_curr == 1):
            self.enumerate_S1(node)
        else:
            self.num_solved += 2
            self.node_counter += 1
            left_node = IterativeNode(node.id, self.node_counter, node.depth + 1, node.C_hat, node.V_hat, node.Vsquare_hat,
                                      node.E_hat, node.s_curr, branch_idx, fixed_in=False, scale_factor=node.scale_factor)
            self.solve_times.append(left_node.solve_time)
            self.node_counter += 1
            right_node = IterativeNode(node.id, self.node_counter, node.depth + 1, node.C_hat, node.V_hat, node.Vsquare_hat,
                                      node.E_hat, node.s_curr, branch_idx, fixed_in=True, scale_factor=node.scale_factor)
            self.solve_times.append(right_node.solve_time)

        # CHECK BOUNDS - need to do something with enumeration?
        # self.z_ub = max{LB, max_L ub(L)}
        # pass in left or right node, not both
        
        return left_node, right_node, right_branch
    
    def enumerate_S0(self, node: IterativeNode):
        for i in arange(node.n_curr):
            x = ones(node.n_curr)
            x[i] = 0
            z = obj_f(x, node.C.Vsquare) + node.scale_factor
            if z > self.z_lb:
                # self.node_counter += 1 
                self.z_lb = z
                # self.optimal_node DEBUG - Need a set solution method
                self.updated_lb_iterations.append(self.total_iterations + 1)


    def enumerate_S1(self, node: IterativeNode):
        for i in arange(node.n_curr):
            x = zeros(node.n_curr)
            x[i] = 1
            z = obj_f(x, node.C.Vsquare) + node.scale_factor
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
            # condition forces initial dive in tree to find a canidate soln
            if self.num_updates < 1: 
                self.right_node_first(left_node, right_node)
            else:
                self.left_node_first(left_node, right_node)
        else:
            if self.num_updates < 1:
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
        # num_updates = len(self.updated_lb_iterations)
        if self.num_updates > 0:
            solved = True
        else:
            solved = False 
        return solved, self.z_lb, self.total_time, self.total_iterations, self.gap, self.num_updates
    
    def set_ub(self, candidate: IterativeNode):
        pass
        # if child node then
        # if candidate.parent_id == self.ub_index
    
    ##### TREE ATTRIBUTES #####

    # def dual_branches