from numpy import (sqrt, matrix, array, dot, diag, eye, flatnonzero, setdiff1d, arange)
from numpy.linalg import (eigh, matrix_rank)
from scipy.linalg import svd
from math import log

def generate_factorizations(C, n, d):
    """Generates the V, Vsquare, and E matrices required for a MESP problem"""
    
    _, s, V = svd(C)
    s[s<1e-6]=0
    
    sqrt_eigen = [0]*n
    for i in range(n):
        if s[i] > 0:
            sqrt_eigen[i] = sqrt(s[i])
        else:
            sqrt_eigen[i] = 0

    V = dot(diag(sqrt_eigen), V)
    V = V[0:d,:]

    V = matrix(V)
    V = V.T

    Vsquare = [[V[i].T * V[i] for i in range(n)]] # V[i] is the row vector
    Vsquare = Vsquare[0]

    E = eye(d, dtype = int)

    return V, Vsquare, E


def obj_f(x, Vsquare):
    """
    Objective function of the MESP
    """

    val = 0.0 # give a more appropriate initial value?
    sel = flatnonzero(x)

    for i in sel:
        val += Vsquare[i]

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

def upd_inv_add(V, E, X, Xs, opti):
    """Update the inverse matrix by adding a rank-one matrix"""
    Y = 0.0
    Ya = 0.0
    Yb = 0.0
    Yc = 0.0
    Ys = 0.0
    
    Y = X + V[opti].T*V[opti]
    
    Ya = Xs*V[opti].T   
    Yb = (E-Xs*X)*V[opti].T
    Yc = 1/(V[opti]*Yb)[0,0]
    Ys = Xs - Yc*(Ya*Yb.T)-Yc*(Yb*Ya.T)+(Yc**2)*(1+(V[opti]*Ya)[0,0])*(dot(Yb,Yb.T))
    
    return Y, Ys

def upd_inv_minus(V, X, Xs, opti):
    """Update the inverse matrix by subtracting a rank-one matrix"""    
    Y = 0.0
    Ya = 0.0
    Yb = 0.0
    Yc = 0.0
    Ys = 0.0
    
    Y = X - V[opti].T*V[opti]
    
    Ya = Xs*V[opti].T
    Yb = Xs*Xs*V[opti].T
    Yc = 1/(Ya.T*Ya)[0,0] 
    Ys = Xs - Yc*(Ya*Yb.T)-Yc*(Yb*Ya.T)+(Yc**2)*(Yb.T*Ya)[0,0]*(Ya*Ya.T)  
    
    return Y, Ys


def srankone(V, E, X, Xs, indexN, n, val):
    """Rank one update for greedy"""
    opti = 0.0
    Y = 0.0
    
    temp = V*(E-Xs*X)*V.T # determinant
    xval = temp.diagonal()
    xval = xval.reshape(n,1)
    xval = list(xval)
    
    maxf=0.0
    opti=0 # add opti to set S
    
    for j in indexN:
        if(xval[j] > maxf):
            maxf = xval[j]
            opti = j
    
    Y,Ys = upd_inv_add(V, E, X,Xs,opti) # update X and Xs
    val = val*maxf # update the objective value
    
    return Y,Ys,opti,val 

def findopt(V, E, X, Xs, i, indexN,n,val):
    """Rank one update for local search"""
    Y=0.0
    Ys=0.0
    
    Y, Ys = upd_inv_minus(V, X,Xs,i)
    
    temp = V*(E-Y*Ys)*V.T
    xval = temp.diagonal()
    xval = xval.reshape(n,1)
    xval = list(xval)
    
    maxf=0.0
    opti=0
    
    for j in indexN:
        if(xval[j]>maxf):
            maxf = xval[j]
            opti = j
    
    val = val+ log(maxf)-log(xval[i])
       
    return Y, Ys, opti, val

def generate_schur_complement_iterative(A, n, selected):
    """
    Assumes len(selected) == 1
    """
    remaining_indices = setdiff1d(arange(n), selected)
    A_shrunk = A[remaining_indices][:, remaining_indices]
    A_left = A[remaining_indices][:, selected]
    selected_val = A[selected, selected]
    if selected_val == 0:
        A_selected_inv = 0
    else:
        A_selected_inv = 1 / selected_val
    A_right = A[selected][:, remaining_indices]

    return A_shrunk - A_left @ (A_selected_inv * A_right)


