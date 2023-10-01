from numpy import (sqrt, matrix, array, dot, diag, eye, flatnonzero)
from numpy.linalg import (eigh, matrix_rank)
from math import log

from mesp.utilities.matrix_computations import generate_factorizations
from mesp.approximation.localsearch import localsearch

class Mesp:
    
    def __init__(self, C: matrix):

        self.C = C

        ## size of the problem
        self.n = C.shape[0]
        self.d = matrix_rank(self.C)

        self.V, self.Vsquare, self.E = generate_factorizations(self.C, self.n, self.d)

    def obj_f(self, x):
        """
        Objective function of the MESP

        :param x: The decision variables
        """
        val = 0.0
        sel = flatnonzero(x)

        for i in sel:
            val += self.Vsquare[i]

        r = matrix_rank(val)
        [a, b] = eigh(val)
        a = a.real # eigenvalues
        b = b.real # eigenvectors

        sorted_a = sorted(a, reverse=True) # sort eigenvalues in decreasing order

        f = 1.0
        for i in range(r):
            f *= sorted_a[i]
        
        if f <= 0:
            print("Singular Matrix")
            return 0

        return log(f)
    
    def solve_approximate(self, s):
        return localsearch(self.V, self.E, self.n, self.d, s)

