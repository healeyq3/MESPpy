from numpy import (sqrt, matrix, array, dot, diag, eye, flatnonzero, argsort)
from numpy.linalg import (eigh, matrix_rank)
from scipy.linalg import svd
from math import log

def find_k(x,s,d):
    sign = 1
    for i in range(s-1):
        k = i
        mid_val = sum(x[j] for j in range(k+1,d))/(s-k-1)
        if mid_val >= x[k+1]-1e-10 and mid_val < x[k]+1e-10:        
            sign = 0
            return k, mid_val
    if sign == 1:
        k = -1
        mid_val = sum(x[j] for j in range(k+1,d))/(s-k-1)
        return k, mid_val
        
def grad_fw(V, Vsquare, S1, S0, x,s,d, varfix=False):
    nx = len(x)
    
    val = 0.0
    
    sel=flatnonzero(x) 
    
    for i in sel:
        val = val+x[i]*Vsquare[i]
        
    [a,b] = eigh(val) 
    a = a.real # engivalues
    b = b.real # engivectors

    sorted_a = sorted(a, reverse=True)     
    k,nu = find_k(sorted_a,s,d)
 
    engi_val = [0]*d
    
    for i in range(d):
        if(a[i] > nu):
            engi_val[i] = 1/a[i]
        else:
            if nu > 0:
                engi_val[i] = 1/nu
            else:
                engi_val[i] = 1/sorted_a[k]
    W = b*diag(engi_val)*b.T # supgradient

    val = 0.0
    subg = [0.0]*nx
    
    temp = V*W*V.T
    val = temp.diagonal()
    val = val.reshape(nx,1)   
    val = list(val)
    
    for i in range(nx):
        subg[i] = val[i][0,0] # supgradient at x
    
    amu = [0.0]*nx
    for i in S1:
        amu[i] = subg[i]

    temp = [0.0]*nx
    for i in range(nx):
        temp[i] = subg[i]
        
    for i in S0:
        temp[i] = min(subg)
    for i in S1:
        temp[i] = min(subg)
        
    temp = array(temp)
    sindex = argsort(-temp)  
    
    # solution of linear oracle maximization
    y = [0]*nx
    for i in S1:
        y[i] = 1
    for i in range(s-len(S1)):
        y[sindex[i]] = 1 

    # construct feasible dual solution and dual gap
    nu = subg[sindex[s-len(S1)-1]]
    mu = [0]*nx
    for i in range(s-len(S1)):
        mu[sindex[i]] = subg[sindex[i]]- nu 
        
    dual_gap = (s-len(S1))*nu+sum(mu)+sum(amu)-s

    f = 1    
    engi_val = sorted(engi_val)
    for i in range(s):
        f = f*engi_val[i] # the objective value of PC
    
    ## Compute the objective difference of original M-DDF and resctricted M-DDF (see Proposition 21 in the paper) 
    fixzero = [0]*nx
    for i in range(nx):
        fixzero[i] = nu + mu[i] - subg[i]
       
    fixone = [0]*nx    
    for i in range(nx):
        fixone[i] = mu[i]
    
    if varfix == False:
        for i in S1:
            fixone[i] = 0.0
            fixzero[i] = 0.0
        for i in S0:
            fixone[i] = 0.0
            fixzero[i] = 0.0
       
    return -log(f), subg, y, dual_gap, fixzero, fixone   
 
##### TO BE DEPRECATED ###### (only difference from usual is at the end it appears)
def grad_fix(V, Vsquare, S1, S0, x,s,d):
    nx = len(x)
    
    val = 0.0
    
    sel=flatnonzero(x) 
    
    for i in sel:
        val = val+x[i]*Vsquare[i]
        
    [a,b] = eigh(val) 
    a = a.real # eigenvalues
    b = b.real # eigenvectors

    sorted_a = sorted(a, reverse=True)     
    k,nu = find_k(sorted_a,s,d)
 
    engi_val = [0]*d
    
    for i in range(d):
        if(a[i] > nu):
            engi_val[i] = 1/a[i]
        else:
            if nu > 0:
                engi_val[i] = 1/nu
            else:
                engi_val[i] = 1/sorted_a[k]
    W = b*diag(engi_val)*b.T # supgradient

    val = 0.0
    subg = [0.0]*nx
    
    temp = V*W*V.T
    val = temp.diagonal()
    val = val.reshape(nx,1)   
    val = list(val)
    
    for i in range(nx):
        subg[i] = val[i][0,0] # supgradient at x
    
    amu = [0.0]*nx
    for i in S1:
        amu[i] = subg[i]

    temp = [0.0]*nx
    for i in range(nx):
        temp[i] = subg[i]
        
    for i in S0:
        temp[i] = min(subg)
    for i in S1:
        temp[i] = min(subg)
        
    temp = array(temp)
    sindex = argsort(-temp)  
    
    # solution of linear oracle maximization
    y = [0]*nx
    for i in S1:
        y[i] = 1
    for i in range(s-len(S1)):
        y[sindex[i]] = 1 

    # construct feasible dual solution and dual gap
    nu = subg[sindex[s-len(S1)-1]]
    mu = [0]*nx
    for i in range(s-len(S1)):
        mu[sindex[i]] = subg[sindex[i]]- nu 
        
    dual_gap = (s-len(S1))*nu+sum(mu)+sum(amu)-s

    f = 1    
    engi_val = sorted(engi_val)
    for i in range(s):
        f = f*engi_val[i] # the objective value of PC
    
    ## Compute the objective difference of original M-DDF and resctricted M-DDF (see Proposition 21 in the paper) 
    fixzero = [0]*nx
    for i in range(nx):
        fixzero[i] = nu + mu[i] - subg[i]
       
    fixone = [0]*nx    
    for i in range(nx):
        fixone[i] = mu[i]
       
    return -log(f), subg, y, dual_gap, fixzero, fixone   