from numpy import (matrix)
from numpy.linalg import (matrix_rank)
from typing import List

from mesp.utilities.matrix_computations import (generate_factorizations, is_psd)

class MespData:
    """
    A Data Wrapper for the positive semidefinite matrix defining the MESP.

    Parameters
    ----------
    C : numpy.matrix
        The positive semidefinite matrix which defines the MESP
    known_psd : bool, optional
        When True, the C matrix is known to be PSD and thus we can directly pass in the
        size and rank associated withe the matrix.
    n : int, optional
        Size of the symmetric matrix (number of cols/rows)
    d : int, optional
        Rank of the matrix
    factorize : bool, optional
        Whether V, Vsquare, and E should be generated for the matrix C. Necessary for the 
        default subproblem bounding algorithm; however, if a user provides their own bounding
        algorithm then these factorizations may not be necessary (thus why generating them is
        optional).
    scale_factor : float, optional
    S1 : List[int], optional
    S0 : List[int], optional
        

    Raises
    ------
    ValueError if the provided C is not PSD

    """
    def __init__(self, C: matrix, known_psd: bool=False, n: int=None, d: int=None,
                 factorize: bool=False, scale_factor: float=0.0,
                 S1: List[int]=[], S0: List[int]=[]):
        # TODO: figure out a way to update V, Vsquare, E when fixing in and out <=> Don't recompute
        if known_psd:
            self.C = C
            self.n = n
            self.d = d
        elif is_psd(C):
            # TODO: any repeated computations here between is_psd and matrix_rank?
            self.C = C
            self.n = C.shape[0]
            self.d = matrix_rank(C)
        else:
            raise ValueError("The provided matrix is not positive semidefinite, and thus cannot be \
                             a covariance matrix. Please provide a different matrix.")
        # TODO: when would a node/subproblem not need these factorizations?
        if factorize:
            self.set_factorizations()
        else:
            self.V, self.Vsquare, self.E = None, None, None
        
        self.S0, self.S1 = S0, S1
        
    def __getattr__(self, item):
        # Make MespData objects behave like typical numpy matrices
        return getattr(self.C, item)
    
    def __repr__(self):
        # Override the representation method
        return repr(self.C)
    
    def set_factorizations(self):
        self.V, self.Vsquare, self.E = generate_factorizations(self.C, self.n, self.d)

    # TODO: do I need these if I'm only going to do iterative computations?
        # => Need for generating solution vector?
    def append_S0(self, idx: int):
        self.S0.append(idx)
    
    def append_S1(self, idx: int):
        self.S1.append(idx)
