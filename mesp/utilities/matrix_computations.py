from numpy import (sqrt, matrix, array, dot, diag, eye, flatnonzero)
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

