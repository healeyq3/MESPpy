import datetime
from numpy import (add, argsort, array)

from mesp.utilities.mesp_data import MespData
from mesp.utilities.grad import (grad_fw as grad, grad_fix)
from mesp.utilities.matrix_computations import (generate_factorizations)
from mesp.approximation.localsearch import localsearch


def frankwolfe(C: MespData, s, varfix=False): 
    
    start = datetime.datetime.now()

    n, d = C.n, C.d
    # S0, S1 = C.S0, C.S1 # not used with shrinking technique 
    S0, S1 = [], []

    V, Vsquare, E = C.V, C.Vsquare, C.E # TODO: this should always work
    
    # if C.V != None:
    #     V, Vsquare, E = C.V, C.Vsquare, C.E
    # else:
    #     V, Vsquare, E = generate_factorizations(C.C, n, d)
    
    # run local search
    LB, x, ltime = localsearch(V, E, n, d, s, S1, S0)
    Obj_f = LB
    # print("The lower bound at current node is", Obj_f)
 
    gamma_t = 0.0  
    t = 0.0
    mindual = 1e+10
    dual_gap = 1 # duality gap
    Obj_f = 1 # primal value
    alpha = 1e-4 # target accuracy
    
    while(dual_gap/(Obj_f+dual_gap) > alpha):
        Obj_f, subgrad, y, dual_gap, w, v = grad(V, Vsquare, S1, S0, x, s, d, varfix)
        
        t = t + 1
        gamma_t = 1/(t+2) # step size
        
        x = [(1-gamma_t)*x_i for x_i in x] 
        y = [gamma_t*y_i for y_i in y]
        x = add(x,y).tolist() # update x
        mindual = min(mindual, Obj_f+dual_gap) # update the upper bound
        
        #print('primal value = ', Obj_f, ', duality gap = ', dual_gap)

        
    # supp = (n-x.count(0)) # size of support of output
    
    end = datetime.datetime.now()
    time = (end-start).total_seconds()

    if not varfix:
        return  mindual, x, time, w, v
    else:
        cut_gap = Obj_f + dual_gap - LB
        return cut_gap, v, w, x, LB


def res_frankwolfe(V, Vsquare, S1, S0, old_x, node_i, node_val, n, d, s): 
    """
    For example, at the last node, we have S1={0}, S0={1} 
    and obtain an optimal solution of the continuous relaxation: old_x
    Next, we will branch the variable x_3, i.e., node_i=3
    At the left branch node, we have S1={0}, S0={1, 3}, i.e., x3=0, node_val=0
    At the right branch node, we have S1={0, 3}, S0={1}, i.e., x3=1, node_val=1
    """

    x = [0]*n  ### a feasible solution transformed from old solution old_x
    for i in range(n):
        x[i] = old_x[i]
    
    if node_val < 0.5:
        index = list(argsort(array(old_x)))
        for i in S1:
            index.remove(i)
        for i in S0:
            index.remove(i)
        mininx = index[0]
      
        x[node_i] = 0
        x[mininx] = x[mininx] + old_x[node_i]
     
    else:
        index = list(argsort(-array(old_x)))
        for i in S1:
            index.remove(i)
        for i in S0:
            index.remove(i)
            
        x[node_i] =  1.0    
        val = max((1.0- old_x[node_i])/2, x[index[0]])
        x[index[0]] = x[index[0]]-val
        x[index[1]] = x[index[1]]-(1-val)
        
    start = datetime.datetime.now()
 
    gamma_t = 0.0  
    t = 0.0
    mindual = 1e+10
    dual_gap = 1 # duality gap
    Obj_f = 1 # primal value
    alpha = 1e-4 # target accuracy
    
    while(dual_gap/(Obj_f+dual_gap) > alpha):
        Obj_f, subgrad, y, dual_gap, w, v = grad(V, Vsquare, S1, S0, x, s, d)
        
        t = t + 1
        gamma_t = 1/(t+2) # step size
        
        x = [(1-gamma_t)*x_i for x_i in x] 
        y = [gamma_t*y_i for y_i in y]
        x = add(x,y).tolist() # update x
        mindual = min(mindual, Obj_f+dual_gap) # update the upper bound

        # print('primal value = ', Obj_f, ', duality gap = ', dual_gap)

        
    # supp = (n-x.count(0)) # size of support of output
    
    end = datetime.datetime.now()
    time = (end-start).total_seconds()

    return  mindual, x, time, w, v

def alter_fw(V, Vsquare, E, fval, S1, S0, n, d, s): 
    
    # run local search
    Obj_f, x, ltime = localsearch(V, E, n, d, s, S1, S0)
    # print("The lower bound at current node is", Obj_f)
 
    gamma_t = 0.0  
    t = 0.0
    mindual = 1e+10
    dual_gap = 1 # duality gap
    Obj_f = 1 # primal value
    alpha = 1e-4 # target accuracy
    
    while(dual_gap/(Obj_f+dual_gap) > alpha):
        Obj_f, subgrad, y, dual_gap, fixzero, fixone = grad(V, Vsquare, S1, S0, x, s, d)
        
        t = t + 1
        gamma_t = 1/(t+2) # step size
        
        x = [(1-gamma_t)*x_i for x_i in x] 
        y = [gamma_t*y_i for y_i in y]
        x = add(x,y).tolist() # update x
        mindual = min(mindual, Obj_f+dual_gap) # update the upper bound
        
        #print('primal value = ', Obj_f, ', duality gap = ', dual_gap)
    cutgap = mindual-fval
    if dual_gap > cutgap + 1e-4:  #talpha: target accuracy
        alpha = 1e-5
        while(dual_gap/(Obj_f+dual_gap) > alpha):
            Obj_f, subgrad, y, dual_gap, fixzero, fixone = grad_fix(V, Vsquare, S1, S0, x, s, d)
            
            t = t + 1
            gamma_t = 1/(t+2) # step size
            
            x = [(1-gamma_t)*x_i for x_i in x] 
            y = [gamma_t*y_i for y_i in y]
            x = add(x,y).tolist() # update x
            mindual = min(mindual, Obj_f+dual_gap) # update the upper bound
            
    return  mindual-fval