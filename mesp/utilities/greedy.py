import math
from numpy import (zeros, flatnonzero)
import datetime

from .matrix_computations import srankone

def grd(V, E, n, d, s, S1=[], S0=[]):
    """
    Function grd needs inputs V, E, n, d, s
    S1 and S0 are optional parameters
    Solution: matrix \sum_{i in S}v_i*v_i^T , its inverse,
    and the running time of the algorithm

    S1 is the subset of selected points
    S0 is the subset of discarded points
    """
    c = 1
    x = [0]*n # chosen set
    for i in S1:
        x[i] = 1
    
    y = [1]*n # unchosen set
    for i in S0:
        y[i] = 0
    for i in S1:
        y[i] = 0
    
    indexN = flatnonzero(y)
    
    index = 0
    X = zeros([d,d])
    Xs = zeros([d,d])
    Y = zeros([d,d])
    Ys = zeros([d,d])
    val = 1 # initial objective value
    fval = 1 
    
    start = datetime.datetime.now()
    
    if len(S1) > 0:
        stop = s + 1 - len(S1)
    else:
        stop = s + 1
    
    while c < stop:       
        Y,Ys,index,fval = srankone(V, E, X,Xs,indexN,n,val)     
        X = Y
        Xs = Ys 
        val = fval
                        
        x[index] = 1
        y[index] = 0
        indexN = flatnonzero(y)   
        c = c + 1
        
    grdx = x # output solution of greedy
    grdf = math.log(val) # output value of greedy
    
    end = datetime.datetime.now()
    time = (end - start).seconds 
    
    return grdf, grdx, Y, Ys, time 