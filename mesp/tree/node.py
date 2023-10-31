from typing import List
from numpy import (matrix, setdiff1d, arange, argmax, concatenate)
from numpy.linalg import slogdet

from mesp.utilities.matrix_computations import (generate_factorizations, generate_schur_complement_iterative, fix_out)
from mesp.bounding.frankwolfe import frankwolfe

class Node:

    branch_idx_constant = None
    C = None
    V = None
    Vsquare = None
    E = None
    n = None
    d = None
    s = None

    def __init__(self, parent_id: int, id: int, depth: int, discarded_points: List[int] = [],
                 selected_points: List[int] = [], fixed_in: bool = None) -> None:
        
        self.parent_id = parent_id
        self.id = id
        self.depth = depth

        self.S0, self.S1 = discarded_points, selected_points
        self.S0_len, self.S1_len = len(discarded_points), len(selected_points)
        self.indices = setdiff1d(arange(Node.n), discarded_points + selected_points) # DEBUG - only compute if needed

        self.n_curr = Node.n - self.S0_len
        self.d_curr = Node.d - self.S0_len
        self.s_curr = Node.s - self.S1_len

        self.is_solved = False
        self.is_integral = False
        self.relaxed_x = None
        self.relaxed_z = float('-inf')
        self.w = None
        self.v = None
        self.backup_branch_idx = None
        self.i_max = None
        self.delta_i_max = None
        self.w_branch = None

    def compute_subproblem_bound(self) -> float:
        obj_val = 0.0
        if not self.fixed_in:
            V_curr, Vsquare_curr, E_curr = self.fix_out(return_full=True)
        if self.fixed_in:
            C_shrunk = self.fix_out()
            C_curr = generate_schur_complement(Node.C, self.S1)
            V_curr, Vsquare_curr, E_curr = generate_factorizations(self.C_curr, Node.n_curr, Node.d_curr)
            Cff = Node.C[self.S1][:, self.S1]
            obj_val += slogdet(Cff)[1]
        
        z, x, time, w, v = frankwolfe(V_curr, Vsquare_curr, E_curr, self.n_curr, self.d_curr, self.s_curr)
        
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
        Note - can stop for-loop after seeing s nonzero values
        """
        for i, x_i in enumerate(self.relaxed_x):
            if int(x_i) != x_i:
                self.is_integral = False
                dist = abs(x_i - self.branch_idx_constant)
                if dist < min_dist:
                    min_dist = dist
                    self.backup_branch_idx = i
        
        return time
    
    def compute_branch_index(self):
        if self.w != None and self.v != None:
            self.i_max = argmax(concatenate((self.w, self.v)))
            if self.i_max < len(self.w):
                self.delta_i_max = self.w[self.i_max]
                self.w_branch = True
            else:
                self.i_max = self.i_max - len(self.w)
                self.delta_i_max = self.v[self.i_max]
                self.w_branch = False
        else:
            self.delta_i_max = 0

    def fix_out(self, return_full = False) -> matrix:
        remaining_indices = setdiff1d(arange(Node.n), self.S0)
        C_curr = Node.C[remaining_indices][:, remaining_indices]
        if return_full:
            V_curr = Node.V[remaining_indices][:, remaining_indices]
            Vsquare_curr = [Node.Vsquare[i] for i in remaining_indices]
            E_curr = Node.E[remaining_indices][:, remaining_indices]
            return 
        
        return C_curr

class IterativeNode:

    branch_idx_constant = None

    def __init__(self, parent_id: int, id: int, depth: int, C: matrix, V: matrix, Vsquare: List[matrix], E: matrix,
                 s: int, branch_idx: int, fixed_in: bool, scale_factor: float = 0.0) -> None:
        
        self.parent_id = parent_id
        self.id = id
        self.depth = depth

        self.n = C.shape[0]
        self.d = self.n
        self.s = s

        self.C = C
        self.V = V
        self.Vsquare = Vsquare
        self.E = E

        # MAke the following more efficient/less gross looking
        if fixed_in and id != 1:
            self.S0 = []
            self.S1 = [branch_idx]
            self.n_curr = self.n - 1
            self.d_curr = self.d - 1
            self.s_curr = self.s - 1
            self.C_hat = C
            self.V_hat = None
            self.Vsquare_hat = None
            self.E_hat = None
        elif not fixed_in and id != 1:
            self.S0 = [branch_idx]
            self.S1 = []
            self.n_curr = self.n - 1
            self.d_curr = self.d - 1
            self.s_curr = self.s 
            self.C_hat = C
            self.V_hat = None
            self.Vsquare_hat = None
            self.E_hat = None
        else:
            self.S0 = []
            self.S1 = []
            self.n_curr = self.n
            self.d_curr = self.d
            self.s_curr = self.s 
            self.C_hat = C
            self.V_hat = V
            self.Vsquare_hat = Vsquare
            self.E_hat = E
            
            
        self.scale_factor = scale_factor
        
        self.is_solved = False
        self.is_integral = False
        self.relaxed_x = None
        self.relaxed_z = float('-inf')
        self.w = None
        self.v = None
        self.backup_branch_idx = None
        self.i_max = None
        self.delta_i_max = None
        self.w_branch = None

    def compute_subproblem_bound(self) -> float:
        if len(self.S0) > 0:
            self.C_hat = fix_out(self.C, self.S0)
            self.V_hat, self.Vsquare_hat, self.E_hat = generate_factorizations(self.C_hat, self.n_curr, self.d_curr)
            # self.C_hat, self.V_hat, self.Vsquare_hat, self.E_hat = self.fix_out_full()
        elif len(self.S1) > 0:
            self.C_hat = generate_schur_complement_iterative(self.C, self.n, self.S1)
            self.V_hat, self.Vsquare_hat, self.E_hat = generate_factorizations(self.C_hat, self.n_curr, self.d_curr)
            Cff = self.C[self.S1][:, self.S1]
            self.scale_factor += slogdet(Cff)[1]

        z, x, time, w, v = frankwolfe(self.V_hat, self.Vsquare_hat, self.E_hat, self.n_curr, self.d_curr, self.s_curr)

        self.relaxed_z = self.scale_factor + z
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
        Note - can stop for-loop after seeing s nonzero values
        """
        for i, x_i in enumerate(self.relaxed_x):
            if int(x_i) != x_i:
                self.is_integral = False
                dist = abs(x_i - IterativeNode.branch_idx_constant)
                if dist < min_dist:
                    min_dist = dist
                    self.backup_branch_idx = i
        
        return time

    ### SEE HERE ###

    # Test the following function

    def fix_out_full(self):
        # print(self.S0) # DEBUG
        remaining_indices = setdiff1d(arange(self.n), self.S0)
        # print(f"Remaining_indices = {remaining_indices}") # DEBUG
        C_hat = self.C[remaining_indices][:, remaining_indices]
        V_hat = self.V[remaining_indices][:, remaining_indices]
        Vsquare_hat = [self.Vsquare[i][remaining_indices][:, remaining_indices] for i in remaining_indices]
        E_hat = self.E[remaining_indices][:, remaining_indices]

        return (C_hat, V_hat, Vsquare_hat, E_hat)
    
    # The following is deprecated (moved to matrix_computations)
    
    # def fix_out_C(self):
    #     remaining_indices = setdiff1d(arange(self.n), self.S0)
    #     # print(f"Remaining_indices = {remaining_indices}") # DEBUG
    #     C_hat = self.C[remaining_indices][:, remaining_indices]
    #     return C_hat

    ### ###
    
    def compute_branch_index(self):
        if self.w != None and self.v != None:
            self.i_max = argmax(concatenate((self.w, self.v)))
            if self.i_max < len(self.w):
                self.delta_i_max = self.w[self.i_max]
                self.w_branch = True
            else:
                self.i_max = self.i_max - len(self.w)
                self.delta_imax = self.v[self.i_max]
                self.w_branch = False
        else:
            self.delta_i_max = 0


