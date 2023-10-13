from typing import List
from numpy import (matrix, slogdet)

from utilities.matrix_computations import (generate_factorizations, obj_f)
from bounding.frankwolfe import frankwolfe

class Node:

    branch_idx_constant = None
    C = None
    n = None
    d = None
    s = None

    def __init__(self, parent_id: int, id: int, depth: int, C_curr: matrix, discarded_points: List[int],
                 selected_points: List[int], V: matrix = None, Vsquare: List[matrix] = None, E: matrix = None) -> None:
        
        self.parent_id = parent_id
        self.id = id
        self.depth = depth

        self.S0, self.S1 = discarded_points, selected_points
        self.S0_len, self.S1_len = len(discarded_points), len(selected_points)

        self.n_curr = Node.n - self.S0_len
        self.d_curr = Node.d - self.S0_len
        self.s_curr = Node.s - self.S1_len

        self.C_curr = C_curr
        if V != None:
            self.V, self.Vsquare, self.E = V, Vsquare, E
        else:
            self.V, self.Vsquare, self.E = None, None, None

        self.is_solved = False
        self.is_integral = False
        self.relaxed_x = None
        self.relaxed_z = float('-inf')
        self.w_dual_variables = None
        self.v_dual_variables = None
        self.i_max = None
        self.backup_branch_idx = None
        self.delta_i_max = None
        self.w_branch = None

    def compute_subproblem_bound(self) -> float:
        obj_val = 0.0
        if self.V == None:
            self.V, self.Vsquare, self.E = generate_factorizations(self.C_curr, Node.n_curr, Node.d_curr)
            Cff = Node.C[self.S1][:, self.S1]
            obj_val += slogdet(Cff)[1]
        
        z, x, time, w, v = frankwolfe(self.V, self.Vsquare, self.E, self.n_curr, self.d_curr, self.s_curr)
        
        # Loop through the relaxed solution and see if it is integral (keep a list which notes the indices being filled in)
        
        self.relaxed_z = obj_val + z
        self.relaxed_x = x
        self.w = w
        self.v = v
        
        self.is_integral = True
        min_dist = 1

        """
        Loop through relaxed solution. If x \in {0, 1}^n then self.is_integral remaines unchanged.
        Otherwise, this node's backup_branch_idx gets set to the index which minimizes the distance in the R sense
        to the chosen branch index constant (usually either 1/2 or 1), which is not binary and which is not already
        branched on.
        """
        for i, x_i in enumerate(self.relaxed_x):
            if int(x_i) != x_i:
                self.is_integral = False
                dist = abs(x_i - self.branch_idx_constant)
                # DEBUG - NEED TO ADJUST INDEXING
                if (i not in self.S0 and i not in self.S1 and
                    dist <  min_dist):
                    min_dist = dist
                    self.backup_branch_idx = i
        
        return time
    
    def compute_branch_index(self):
        # if w != None and v != None
        pass
