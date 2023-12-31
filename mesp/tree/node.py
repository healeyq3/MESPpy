from typing import List
from numpy import (matrix, setdiff1d, arange, argmax, concatenate, array, ndarray)
from numpy.linalg import slogdet

from mesp.utilities.mesp_data import MespData
from mesp.utilities.matrix_computations import (generate_factorizations, generate_schur_complement_iterative, fix_out)
from mesp.bounding.bound_chooser import BoundChooser
    
class IterativeNode:
    """
    Notes
    -----
    The relaxed value associated with a node DOES include the scale_factor
    
    """

    branch_idx_constant = None # used for default branching
    bound_chooser: BoundChooser = None

    def __init__(self, parent_id: int, id: int, depth: int, C: MespData,
                 s: int, branch_idx: int, fixed_in: bool) -> None:
        
        self.parent_id : int = parent_id
        self.id: int = id
        self.depth: int = depth

        if not fixed_in and id != 1:
            C_hat = fix_out(C.C, [branch_idx])
            self.C = MespData(C_hat, known_psd=True, n=C.n-1, d=C.d-1, factorize=True,
                              scale_factor=C.scale_factor)
            self.C.append_S0(branch_idx)
            self.s = s
        
        elif fixed_in and id != 1:
            C_hat = generate_schur_complement_iterative(C.C, C.n, [branch_idx])
            # C_orig = C.C
            C_ff = C.C[branch_idx][:, branch_idx]
            scale_factor = slogdet(C_ff)[1]
            self.C = MespData(C_hat, known_psd=True, n=C.n-1, d=C.d-1, factorize=True,
                              scale_factor=C.scale_factor + scale_factor)
            self.C.append_S1(branch_idx)
            self.s = s - 1
        
        else: # root node
            self.C = C
            self.s = s      
        
        self.solve_time = self.compute_bound()
        self.integral = self.is_integral()
        
        self.i_max = None
        self.delta_i_max = None
        self.w_branch = None
    
    def compute_bound(self) -> float:

        bound_algorithm = IterativeNode.bound_chooser.get_bound(self.s)
        bound_object = bound_algorithm(self.C, self.s)

        self.relaxed_z = self.C.scale_factor + bound_object[0]
        self.relaxed_x: ndarray = array(bound_object[1])
        
        if len(bound_object) == 5:
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
            

