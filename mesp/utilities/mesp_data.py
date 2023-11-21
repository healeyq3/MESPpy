from numpy import (matrix)
from numpy.linalg import (matrix_rank)

from mesp.utilities.matrix_computations import (generate_factorizations, is_psd)

class MespData:

    def __init__(self, C: matrix, factorize: bool=False):
        # Need to handle C already known psd
        if is_psd(C):
            self.C = C
            self.n = C.shape[0]
            self.d = matrix_rank(C)
        else:
            raise ValueError("The provided matrix is not positive semidefinite, and thus cannot be \
                             a covariance matrix. Please provide a different matrix.")
        if factorize:
            self.V, self.Vsquare, self.E = generate_factorizations(self.C, self.n, self.d)
        else:
            self.V, self.Vsquare, self.E = None, None, None
        
        self.S0, self.S1 = [], []
        self.branch_idx = None
        
    def __getattr__(self, item):
        # Make MespData objects behave like typical numpy matrices
        return getattr(self.C, item)
    
    def __repr__(self):
        # Override the representation method
        return repr(self.C)