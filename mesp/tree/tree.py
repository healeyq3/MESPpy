from numpy import (array, ndarray, zeros, ones, arange, setdiff1d, where, isin, append)
from typing import Tuple, List
import numbers
import datetime
import time

from mesp.utilities.mesp_data import MespData
from mesp.utilities.matrix_computations import (generate_factorizations, obj_f)
from mesp.tree.node import IterativeNode
from mesp.bounding.bound_chooser import BoundChooser

class Tree:
    """
    Branch and Bound tree to exactly solve the MESP.

    Parameters
    ----------
    C: MespData
        The covariance matrix and associated objects which are used at the root of the tree.
    s : int
        The size of the selected subset the MESP is being solved for
    optimal_approx : float
        The best known lower bound for the MESP with a given n and s
    branching_idx_constant : float, optional
        When a subproblem needs to be branched, the chosen branching index (if the dual branching strategy is not used)
        will be argmin_i |x_i - branching_idx_constant| where x is the relaxed solution of a given subproblem.
    epsilon : number, optional
        Numerical error parameter. Used to instantiate the tree's lower bound (z_LB = z_hat - epsilon) with the goal
        of forcing the tree to generate a solution.
    """

    def __init__(self, C: MespData, s: int, optimal_approx: float, bound_chooser: BoundChooser,
                 branch_idx_constant: float = 0.5, epsilon: float = 1e-6) -> None:
        
        # Parameter assignments
        self.C : MespData = C
        self.s = s
        self.EPS = epsilon # Numerical Error Term

        # Assign Node Class attributes
        IterativeNode.branch_idx_constant = branch_idx_constant
        IterativeNode.bound_chooser = bound_chooser
    
        # note the branch index and fixed_in params are ignored for root
        # Also note that the time it takes to bound the roote node is not 
        # added to the tree's solution time, but instead is tracked in the
        # MESP compilation time (however, the upper bounding time
        # can be accessed in solve_times)
        root : IterativeNode = IterativeNode(0, 1, 1, C, s, 0, False) 
        self.open_nodes = [root]

        self.z_hat = optimal_approx + C.scale_factor
        self.z_lb = optimal_approx + C.scale_factor - epsilon

        self.z_inital_bound = root.relaxed_z
        self.z_lub = self.z_inital_bound
        self.z_ub = self.z_inital_bound
        self.ub_id = 1 
        self.gap = self.z_inital_bound - self.z_hat
        self.delta_criterion = abs(self.gap) / 2

        ### Framework Statistics ###
        self.node_counter = 1
        self.log = "" # TODO: rebuild logging functionality - see CVXPY for proper approach
        self.total_time = root.solve_time # total time it takes to solve the tree
        self.solve_times = [root.solve_time]
        self.total_iterations = 1 
        self.num_solved = 1 # the number of times a subproblem's bound had to be computed
        self.updated_lb_iterations = [] # iterations which corresponded to z_lb being updated
        self.dual_branched = [] # iterations which branched using the dual variables
        self.num_updates = 0
        self.optimal_C : MespData = None
        self.optimal_x: ndarray = None # |optimal_x| = n_hat <= doesn't include fixed indices
        self.solution: ndarray = zeros(C.n) # |solution| = n
        self.time_to_opt = None

        self.solved = False
        self.approximate_verified = False 
        self.early_termination = False
    
    ### End of Constructor ###

    ### Begin methods used for solving tree ###

    def solve_tree(self, tol: float=1e-5, timeout: float=0.0) -> Tuple[bool, ]:
        """

        Parameters
        ----------
        tol : float
            How close (as measured by Hamming distance) the tree's upper bound should be to 
            the approximate solution OR the lower bound to guarantee solution optimality.
        timeout : float TODO: this is not currently implemented: timing out is not allowed right now
            How many minutes the B&B tree will be allowed to add new nodes to its open queue. 
            After the timeout, the nodes in the queue will be evaluated without 
            Pass in 0 if you do not wish for there to be a timeout.

        Returns
        -------
        bool : 
            Whether the tree *generated* a new solution
        bool : 
            Whether the tree guarantees an optimal solution
        float :
            max{z_LB, z_hat}
        float : 
            runtime, in seconds


        Notes
        -----
        TODO: Look through these
        opt_found: We have a solution whose associated value is within our tolerance. Two cases
            1. z_UB - z_hat < tol and z_LB < z_hat : in this case, the subproblem with an associated 
                continuous relaxation value which is at most tol larger than z_hat
                in this case, the subproblem with the largest continuous relaxation value is at most tol 
                larger than z_hat, which implies that z_hat ~ z^*
            2. z_UB - z_LB < tol and z_LB > z_hat + eps : in this case we have found a candidate solution which has a larger
                value than z_hat, thus z_hat not ~ z^*. However, once z_UB is within tol of z_LB, we 
                accept z_LB ~ z^*

        Algorithm:

        While Q not empty and z_UB > z_LB, pop node
            check_optimality() => 1 and 2 above

            Evaluating a Node (i)
            1. if z_i < z_LB => prune
            2. if z_i > z_LB and x_i not in {0, 1}^n_hat 
                => branch
            3. if z_i > z_LB and x_i in {0, 1}^n_hat (it is possible that z_LB = z^UB,
            how to check this? z_UB would have been updated on the creation of the branch)
                3a. if z_i > z_hat

        TRACK SIZE OF QUEUE <- Interesting stat to see if the UB allows termination
            to occur before queue fully enumerated.
        
        """

        self.TIMEOUT = timeout

        solve_time_start = time.time()

        while self.open_nodes and self.z_ub > self.z_lb:

            iteration_start = time.time()

            node: IterativeNode = self.open_nodes.pop()

            self.solve_times.append(node.solve_time)
            
            self.evaluate_node(node, solve_time_start)

            self.check_optimality(tol, solve_time_start)

            self.total_iterations += 1
            iteration_length = time.time() - iteration_start
            self.total_time += iteration_length

            print(f"Iteration {self.total_iterations} | current LB = {self.z_lb} | Number of Open Subproblems = {len(self.open_nodes)}"
                + f" | Total Running Time = {self.total_time} seconds ", end = "\r") 
            
        self.total_time = time.time() - solve_time_start
        return self.evaluate_tree()
           
    def check_optimality(self, tol: float, curr_runtime):
        # print('z_UB: ', self.z_ub)
        if (self.z_lb <= self.z_hat + self.EPS
            and abs(self.z_ub - self.z_hat) < tol):
            # print('CONDITION 1 IN OPT CHECK') # DEBUG
            self.time_to_opt = time.time() - curr_runtime
            self.approximate_verified = True
        elif (self.z_lb > self.z_hat + self.EPS
              and abs(self.z_ub - self.z_lb) < tol):
            # print('CONDITION 2 IN OPT CHECK') # DEBUG
            self.time_to_opt = time.time() - curr_runtime
            self.approximate_verified = False

    def evaluate_node(self, node: IterativeNode, runtime_start):

        z = node.relaxed_z
        is_integral = node.integral

        if z >= self.z_lb and is_integral:
            self.z_lb = z
            self.optimal_C = node.C
            self.optimal_x = node.relaxed_x
            self.updated_lb_iterations.append(self.total_iterations + 1)
            self.num_updates += 1
        
        elif z > self.z_lb and not is_integral:
            curr_time = time.time() - runtime_start

            if self.TIMEOUT == 0 or self.TIMEOUT * 60 > curr_time:
                node.compute_branch_index() # only compute branch index if need to
                left_node, right_node, right_branch = self.branch(node)
                self.add_nodes(left_node, right_node, right_branch)
            else:
                self.early_termination = True
                # TODO: implement early termination enumeration
            
            if left_node != None and left_node.parent_id == self.ub_id: self.set_ub() # WLOG can choose right_node

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

        if (node.s== node.C.n - 1):
            self.enumerate_S0(node)
        elif (node.s == 1):
            self.enumerate_S1(node)
        else:
            # print("non enumerate branch") # DEBUG
            self.num_solved += 2 # Since nodes are bounded upon instantiation
            self.node_counter += 1
            left_node = IterativeNode(node.id, self.node_counter, node.depth + 1, node.C, node.s, branch_idx, fixed_in=False)
            self.solve_times.append(left_node.solve_time)
            self.node_counter += 1
            right_node = IterativeNode(node.id, self.node_counter, node.depth + 1, node.C, node.s, branch_idx, fixed_in=True)
            self.solve_times.append(right_node.solve_time)
        
        return left_node, right_node, right_branch
    
    def enumerate_S0(self, node: IterativeNode):
        for i in arange(node.C.n):
            x = ones(node.C.n)
            x[i] = 0
            z = obj_f(x, node.C.Vsquare) + node.C.scale_factor
            if z > self.z_lb:
                self.optimal_C = node.C
                self.optimal_x = x
                self.z_lb = z
                self.num_updates += 1
                self.updated_lb_iterations.append(self.total_iterations + 1)


    def enumerate_S1(self, node: IterativeNode):
        for i in arange(node.C.n):
            x = zeros(node.C.n)
            x[i] = 1
            z = obj_f(x, node.C.Vsquare) + node.C.scale_factor
            if z > self.z_lb:
                self.optimal_C = node.C
                self.optimal_x = x
                self.z_lb = z
                self.num_updates += 1
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
        if self.num_updates > 0:
            self.solved = True
        else:
            self.solved = False 
        
        self.generate_solution_vector()

        # TODO: The second argument needs to be changed once timeouts are allowed
        # <=> right now if the solve function terminates without ctrl+C then the conditions of the
        # while condition have been met, thus we've verified an optimal solution.
        return self.solved, True, max(self.z_lb, self.z_hat), self.solution, self.total_time
    
    def set_ub(self):
        # print('in set ub') # DEBUG
        # Can use structure of tree to make this more efficient in future
        self.z_ub = self.open_nodes[0].relaxed_z
        self.ub_id = self.open_nodes[0].id
        for node in self.open_nodes:
            if node.relaxed_z > self.z_ub:
                self.z_ub = node.relaxed_z
                self.ub_id = node.id

    def generate_solution_vector(self):

        # TODO: abstract the indexing code found here, in MespData, and in VariableFixing
        # to one central place

        all_indices = arange(self.C.n)

        remaining_indices = setdiff1d(all_indices, self.optimal_C.S0 + self.optimal_C.S1)
    
        for i in all_indices:
            if i in self.optimal_C.S0:
                self.solution = append(self.solution, [0])
            elif i in self.optimal_C.S1:
                self.solution = append(self.solution, [1])
            else:
                relative_index = where(isin(remaining_indices, [i]))[0][0]
                self.solution = append(self.solution, [self.optimal_x[relative_index]])