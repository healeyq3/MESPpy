import datetime
from numpy import (array, argsort, add, matrix, ndarray, setdiff1d, where, isin, arange)
from numpy.linalg import slogdet
from typing import Tuple
import time

from mesp.utilities.mesp_data import MespData
from mesp.utilities.matrix_computations import (fix_out, generate_schur_complement)
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

# ========== DEPRECATE ==========

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

# ==========       ============

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

def fix_variables(s: int, C: MespData) -> Tuple[bool, MespData, int, float]:
    """
    
    Returns
    -------
    bool : whether variables were successfully fixed
    MespData: if there are fixed variables, the corresponding new data object
    int : if variables are fixed, the corresponding number of variables which still need to be chosen
    float : the time it took to peform the variable fixing operation
    """
    start_time = time.time()
    S1, S0, _  = varfix(C.V, C.Vsquare, C.E, C.n, C.d, s)
    if S1 == 0 and S0 == 0:
        end_time = time.time() - start_time
        return False, C, s, end_time
    else:
        C_hat = C.C
        n_hat = C.n
        d_hat = C.d
        s_hat = s
        scale_factor = 0
        # There are variables which can be fixed out of the problem.
        if len(S0) > 0:
            C_hat = fix_out(C.C, S0)
            n_hat = n_hat - len(S0)
            d_hat = d_hat - len(S0)
        # There are variables which can be fixed into the problem
        if len(S1) > 0:
            C_ff = C.C[S1][:, S1]
            scale_factor += slogdet(C_ff)[1]
            """
            The following logic is:
            remaining_indices = [n] \ S0
            updated_indices contains the indices in remaining_indices
                where the values in S1 match with the values in
                remaining_indices
            Example (using 1-indexing): n=9, S0 = [1, 3, 5, 8], S1 = [2, 7]
                [n] \ S0 = [2, 4, 6, 7, 9]
                S1_bar = [1, 4]
            """
            remaining_indices = setdiff1d(arange(C.n), S0)
            updated_indices = where(isin(remaining_indices, S1))[0]
            C_hat = generate_schur_complement(C_hat, n_hat, updated_indices)
            n_hat = n_hat - len(S1)
            s_hat = s - len(S1)
            d_hat = d_hat - len(S1)
        C_hat = MespData(C_hat, known_psd=True, n=n_hat, d=d_hat, factorize=True,
                         scale_factor=scale_factor, S1=S1, S0=S0)
        end_time = time.time() - start_time
        return True, C_hat, s_hat, end_time
    

############### TO BE REMOVED #####################

# def fix_variables(self, s: int) -> Tuple[bool, matrix, int, int, int, float]:
#     S1, S0, _ = varfix(self.V, self.Vsquare, self.E, self.n, self.d, s)
#     if S1 == 0 and S0 == 0:
#         print("No variables could be fixed")
#         return False, None, None, None, None, None
#     else:
#         C_hat = self.C
#         n_hat = self.n
#         d_hat = self.d
#         s_hat = s
#         scale_factor = 0
#         if len(S0) > 0:
#             C_hat = fix_out(self.C, S0)
#             n_hat = n_hat - len(S0)
#             d_hat = d_hat - len(S0)
#         if len(S1) > 0:
#             remaining_indices = setdiff1d(arange(self.n), S0)
#             updated_indices = where(isin(remaining_indices, S1))[0]
#             #### Scaling ###
#             C_ff = C_hat[updated_indices][:, updated_indices]
#             scale_factor += slogdet(C_ff)[1]
#             ### ###
#             C_hat = generate_schur_complement(C_hat, n_hat, updated_indices)
#             n_hat = n_hat - len(S1)
#             s_hat = s_hat - len(S1)
#             d_hat = d_hat - len(S1)
#         return True, C_hat, n_hat, d_hat, s_hat, scale_factor
