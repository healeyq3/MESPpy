import datetime
from numpy import (array, argsort, add)

from mesp.bounding.frankwolfe import (frankwolfe, alter_fw)
from mesp.approximation.localsearch import localsearch
from mesp.utilities.grad import grad_fix

def frankwolfelocal(V, Vsquare, E, n, d, s): 
    
    start = datetime.datetime.now()
    
    # run local search
    LB, x, ltime = localsearch(V, E, n, d, s)
    Obj_f = LB
    # print("The lower bound at current node is", Obj_f)
 
    gamma_t = 0.0  
    t = 0.0
    mindual = 1e+10
    dual_gap = 1 # duality gap
    Obj_f = 1 # primal value
    alpha = 1e-5 # target accuracy
    
    cut_gap = 0.0 # the gap to derive optimality cuts    
    xsol = [2]*n
    S1 = []  # store selected points
    S0 = []  # store discarded points    
        
    while(dual_gap/(Obj_f+dual_gap) > alpha):
        Obj_f, subgrad, y, dual_gap, fixzero, fixone = grad_fix(V, Vsquare, S1, S0, x, s, d)
        
        t = t + 1
        gamma_t = 1/(t+2) # step size
        
        x = [(1-gamma_t)*x_i for x_i in x] 
        y = [gamma_t*y_i for y_i in y]
        x = add(x,y).tolist() # update x
        mindual = min(mindual, Obj_f+dual_gap) # update the upper bound
        
        #print('primal value = ', Obj_f, ', duality gap = ', dual_gap)
        
        ## derive optimality cuts
        cut_gap =  Obj_f + dual_gap - LB
        for i in range(n):
            if cut_gap < fixzero[i]:  # restricted DDF < DDF if i-th point is selected; Hence, discard i-th point                
                xsol[i] = 0
            if cut_gap < fixone[i]:  # restricted DDF < DDF if i-th point is discarded; Hence, select i-th point                 
                xsol[i] = 1
        
        S0 = [i for i in range(n) if xsol[i] == 0] # discarded points
        S1 = [i for i in range(n) if xsol[i] == 1] # selected points

    
    end = datetime.datetime.now()
    time = (end-start).seconds

    return  S1, S0, fixone, fixzero, cut_gap, x, LB

def cut_gap_fixing(V, Vsquare, E, n, d, s):
    cut_gap, v, w, x, LB = frankwolfe(V, Vsquare, E, n, d, s, varfix=True)

    xsol = [2]*n
    S1 = []
    S0 = []

    # Derive optimality cuts
    for i in range(n):
        if cut_gap < w[i]: # restricted DDF < DDF if i-th point is selected; Hence, discard i-th point
            xsol[i] = 0 
        if cut_gap < v[i]: # restricted DDF < DDF if i-th point is discarded; Hence, select i-th point 
            xsol[i] = 1 
    S0 = [i for i in range(n) if xsol[i] == 0] # discarded points
    S1 = [i for i in range(n) if xsol[i] == 1] # selected points

    return S1, S0, v, w, cut_gap, x, LB

### output S1, S0, time
### S1: indices of variables fixed to be 1 
### S0: indices of variables fixed to be 0
def varfix(V, Vsquare, E, n, d, s):
    start = datetime.datetime.now()

    S1, S0, fixone, fixzero, init_cut_gap, cxsol, fval = frankwolfelocal(V, Vsquare, E, n, d, s)

    #### derive cut (b) ####
    indexS = []
    indexT = S0
    temp = array(fixzero)
    sortinx = argsort(-temp)
    setzero = list(set(range(n))-set(S0))
    cutgap = -1
    for i in range(n):
        if temp[sortinx[i]] < init_cut_gap and sortinx[i] in setzero:
            indexS= []
            indexS = S1
            indexS.append(sortinx[i]) # suppose select i-th point 
            cutgap = alter_fw(V, Vsquare, E, fval, indexS, indexT, n, d, s)
            # print(i, cutgap)
            if cutgap < 0: # restricted DDF < DDF if i-th point is selected; Hence, discard i-th point
                S0.append(sortinx[i])
            indexS.remove(sortinx[i])
            
        if cutgap > 1e-2:
            break    
        
    #### derive cut (a) ### 
    indexS = []
    indexT = []

    if s>=20:
        sone = [i for i in range(n) if cxsol[i] >= 0.8]
    else:
        sone = [i for i in range(n) if cxsol[i] >= 0.5]
    sone = list(set(sone)-set(S1))
    cutgap = -1
    for i in sone:
        indexT = []
        indexT = S0
        indexT.append(i) # suppose discard i-th point 
        
        cutgap = alter_fw(V, Vsquare, E, fval, indexS, indexT, n, d, s)

        if cutgap < 0: # restricted DDF < DDF if i-th point is discarded; Hence, select i-th point
            S1.append(i)
            
        indexT.remove(i)

    
    print("The number of variables fixed being 1 is", len(S1))
    print("The number of variables fixed being 0 is", len(S0))
    
    end = datetime.datetime.now()
    time = (end-start).total_seconds()
    
    return S1, S0, time