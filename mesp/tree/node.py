from typing import List
from numpy import (matrix, setdiff1d, arange, argmax, concatenate)
from numpy.linalg import slogdet

from mesp.utilities.mesp_data import MespData
from mesp.utilities.matrix_computations import (generate_factorizations, generate_schur_complement_iterative, fix_out)
from mesp.bounding.bound_chooser import BoundChooser

### TODOS ###
'''
1. Turn existing Node class into "BatchNode"
2. Create a new "Node" class to serve as a parent for all other types of nodes.
    This class will have the ids, depth, class constants (like bound chooser), and
    functions
'''


# class Node:

#     branch_idx_constant = None
#     C = None
#     V = None
#     Vsquare = None
#     E = None
#     n = None
#     d = None
#     s = None
#     choose_bound = None

#     # def __init__(self, parent_id: int, depth: int, C: matrix, discarded_points: List[int]=[], selected_points: List[int]=[],
#     #              )

#     def __init__(self, parent_id: int, id: int, depth: int, discarded_points: List[int] = [],
#                  selected_points: List[int] = [], fixed_in: bool = None) -> None:
        
#         self.parent_id = parent_id
#         self.id = id
#         self.depth = depth

#         self.S0, self.S1 = discarded_points, selected_points
#         self.S0_len, self.S1_len = len(discarded_points), len(selected_points)
#         self.indices = setdiff1d(arange(Node.n), discarded_points + selected_points) # DEBUG - only compute if needed

#         self.n_curr = Node.n - self.S0_len
#         self.d_curr = Node.d - self.S0_len
#         self.s_curr = Node.s - self.S1_len

#         self.is_solved = False
#         self.is_integral = False
#         self.relaxed_x = None
#         self.relaxed_z = float('-inf')
#         self.w = None
#         self.v = None
#         self.backup_branch_idx = None
#         self.i_max = None
#         self.delta_i_max = None
#         self.w_branch = None

#     def compute_subproblem_bound(self) -> float:
#         obj_val = 0.0
#         if not self.fixed_in:
#             V_curr, Vsquare_curr, E_curr = self.fix_out(return_full=True)
#         if self.fixed_in:
#             C_shrunk = self.fix_out()
#             C_curr = generate_schur_complement(Node.C, self.S1)
#             V_curr, Vsquare_curr, E_curr = generate_factorizations(self.C_curr, Node.n_curr, Node.d_curr)
#             Cff = Node.C[self.S1][:, self.S1]
#             obj_val += slogdet(Cff)[1]
        
#         z, x, time, w, v = frankwolfe(V_curr, Vsquare_curr, E_curr, self.n_curr, self.d_curr, self.s_curr)
        
#         self.relaxed_z = obj_val + z
#         self.relaxed_x = x
#         self.w = w
#         self.v = v
        
#         self.is_integral = True
#         min_dist = 1

#         """
#         Loop through relaxed solution. If x \in {0, 1}^n then self.is_integral remaines unchanged.
#         Otherwise, this node's backup_branch_idx gets set to the index which minimizes the distance in the R sense
#         to the chosen branch index constant (usually either 1/2 or 1), which is not binary and which is not already
#         branched on.
#         Note - can stop for-loop after seeing s nonzero values
#         """
#         for i, x_i in enumerate(self.relaxed_x):
#             if int(x_i) != x_i:
#                 self.is_integral = False
#                 dist = abs(x_i - self.branch_idx_constant)
#                 if dist < min_dist:
#                     min_dist = dist
#                     self.backup_branch_idx = i
        
#         return time
    
#     def compute_branch_index(self):
#         if self.w != None and self.v != None:
#             self.i_max = argmax(concatenate((self.w, self.v)))
#             if self.i_max < len(self.w):
#                 self.delta_i_max = self.w[self.i_max]
#                 self.w_branch = True
#             else:
#                 self.i_max = self.i_max - len(self.w)
#                 self.delta_i_max = self.v[self.i_max]
#                 self.w_branch = False
#         else:
#             self.delta_i_max = 0

#     def fix_out(self, return_full = False) -> matrix:
#         remaining_indices = setdiff1d(arange(Node.n), self.S0)
#         C_curr = Node.C[remaining_indices][:, remaining_indices]
#         if return_full:
#             V_curr = Node.V[remaining_indices][:, remaining_indices]
#             Vsquare_curr = [Node.Vsquare[i] for i in remaining_indices]
#             E_curr = Node.E[remaining_indices][:, remaining_indices]
#             return 
        
#         return C_curr

class NonShrinkingNode(Node):
    pass
    # def __init__(self, parent_id: int, depth: int, discarded_points: List[int]=[])

    
class IterativeNode:

    # What can be abstracted to a super node class? Then have iterative node and non-shrinking node
    # > ids, depth
    # > branch_idx_constant and bound chooser can also be super node class attributes
    # > all functions

    branch_idx_constant = None
    bound_chooser: BoundChooser = None

    def __init__(self, parent_id: int, id: int, depth: int, C: MespData,
                 s: int, branch_idx: int, fixed_in: bool, scale_factor: float = 0.0) -> None:
        
        self.parent_id : int = parent_id
        self.id: int = id
        self.depth: int = depth

        if not fixed_in and id != 1:
            C_hat = fix_out(C.C, [branch_idx])
            self.C = MespData(C_hat, known_psd=True, n=C.n-1, d=C.d-1, factorize=True)
            self.C.append_S0(branch_idx)
            self.s = s
        
        elif fixed_in and id != 1:
            C_hat = generate_schur_complement_iterative(C.C, C.n, [branch_idx])
            C_orig = self.C.C
            self.C = MespData(C_hat, known_psd=True, n=C.n-1, d=C.d-1, factorize=True)
            self.C.append_S1(branch_idx)
            C_ff = C_orig[branch_idx][:, branch_idx]
            self.scale_factor = scale_factor + slogdet(C_ff)[1]
            self.s = s - 1
        
        else: # root node
            self.C = C
            self.s = s      
        
        self.solve_time = self.compute_bound()
        self.integral = self.is_integral()
        
        self.backup_branch_idx = None
        self.i_max = None
        self.delta_i_max = None
        self.w_branch = None
    
    def compute_bound(self) -> float:

        bound_algorithm = IterativeNode.bound_chooser.get_bound(self.s)
        bound_object = bound_algorithm(self.C, self.s)

        self.relaxed_z = self.scale_factor + bound_object[0]
        self.relaxed_x = bound_object[1]
        
        if len(bound_object == 5):
            self.w = bound_object[3]
            self.v = bound_object[4]
        else:
            self.w = None
            self.v = None
        
        return bound_object[2]

    
    def is_integral(self) -> bool:
        """
        Loop through relaxed solution. If x \in {0, 1}^n then self.is_integral remaines unchanged.
        Otherwise, this node's backup_branch_idx gets set to the index which minimizes the distance in the R sense
        to the chosen branch index constant (usually either 1/2 or 1), which is not binary and which is not already
        branched on.
        Note - can stop for-loop after seeing s nonzero values
        """
        
        is_integral = True
        min_dist = 1
        
        for i, x_i in enumerate(self.relaxed_x):
            if int(x_i) != x_i:
                is_integral = False
                dist = abs(x_i - IterativeNode.branch_idx_constant)
                if dist < min_dist:
                    min_dist = dist
                    self.backup_branch_idx = i
        return is_integral

    def compute_branch_index(self):
        if self.w != None and self.v != None:
            w_i_max = argmax(self.w)
            w_max = self.w[w_i_max]
            
            v_i_max = argmax(self.v)
            v_max = self.v[v_i_max]

            if w_max > v_max:
                self.i_max = w_i_max
                self.delta_i_max = w_max
                self.w_branch = True
            else:
                self.i_max = v_i_max
                self.delta_i_max = v_max
                self.w_branch = False
        else:
            self.delta_i_max = 0
            

