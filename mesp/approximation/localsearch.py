import datetime
from typing import Tuple
from numpy import flatnonzero, ndarray

from mesp.utilities.matrix_computations import (findopt, upd_inv_add)
from mesp.utilities.greedy import grd


def localsearch(V, E, n, d, s, S1=[], S0=[]) -> Tuple[float, ndarray, float]:  
    """
    Function localsearch needs input n, d and s; outputs the objectve value,
    Inputs S1 and S0 are optional 
    solution, matrix \sum_{i in S}v_i*v_i^T and its inverse, 
    and running time of the local search algorithm

    Let subset $S_1$ denote selected points
    Let subset $S_0$ denote discarded points
    """  
    start = datetime.datetime.now()
    
    ## greedy algorithm
    # bestf, bestx, X, Xs, gtime = grd(V, E, n, d, s, S1, S0)
    bestf, bestx, X, Xs, gtime = grd(V, E, n, d, s)
    # print("The running time of Greedy algorithm = ", gtime)
    #print('The current objective value is:', bestf) # the objective value
    
    sel = [i for i in range(n) if bestx[i] == 1] # chosen set
    if len(S1) > 0:
        sel = list(set(sel) - set(S1))
    t = [i for i in range(n) if bestx[i] == 0] # unchosen set 
    if len(S0) > 0:
        t = list(set(t) - set(S0))                 
  
    ## local search    
    Y = 0.0
    Ys = 0.0 
    fval = 0.0
    optimal = False

    while(optimal == False):
        optimal = True
        
        for i in sel :
            Y, Ys, index,fval = findopt(V, E, X, Xs, i, t, n, bestf)

            
            if fval > bestf:
                optimal = False                
                bestx[i] = 0
                bestx[index] = 1 # update solution                 
                bestf = fval # update the objective value
                #print('The current objective value is:', bestf)
                
                X, Xs = upd_inv_add(V, E, Y, Ys, index) # update the inverse
                
                sel.remove(i)
                sel.append(index)# update chosen set
                t.remove(index)
                t.append(i)# update the unchosen set
                
                break

    end = datetime.datetime.now()
    time = (end - start).total_seconds()         
    #print('The final objective value is:', bestf)
       
    return bestf, bestx, time
