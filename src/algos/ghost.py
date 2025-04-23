from typing import Tuple,Callable, List
import pandas as pd
import torch.utils.data.dataloader
import numpy as np
import scipy as sp
from copy import deepcopy
from scipy.optimize import linprog
from qpsolvers import solve_qp
import autoray as ar
import timeit
from itertools import cycle
from .utils import net_grads_to_tensor, net_params_to_tensor
import cvxpy as cp
import torch
        

def computekappa(cval, cgrad, lamb, rho, mc, n, scalef):  
    obj = np.concatenate(([1.], np.zeros((n,))))
    Aubt = np.column_stack((-np.ones(mc), np.array(cgrad)))
    c_viol = [np.maximum(cv, 0) for cv in cval]
    max_c_viol = np.max(c_viol)
    try:
       res = linprog(c=obj, A_ub=Aubt, b_ub=-c_viol, bounds=[(-rho, rho)])
       return (1-lamb)*max_c_viol + lamb*max(0, res.fun)
    except:
       return (1-lamb)*max_c_viol + lamb*max(0, rho)
   
   
def __computekappa__(cval, cgrad, lamb, rho, mc, n,):  
    # Objective: minimize t (first variable) with [t; d] as variables
    obj = np.concatenate(([1.], np.zeros(n)))
    
    # Constraints: -t + ∇c_i^T d ≤ -c_i  ⟺  [-1, ∇c_i] • [t; d] ≤ -c_i
    Aubt = np.column_stack((-np.ones(mc), cgrad))  # Shape (mc, n+1)
    b_ub = -np.array(cval)  # Use original cval (not max(cval, 0))
    
    # Bounds: t is unbounded, d ∈ [-ρ, ρ] (infinity-norm)
    bounds = [(None, None)] + [(-rho, rho) for _ in range(n)]
    
    try:
        res = linprog(c=obj, A_ub=Aubt, b_ub=b_ub, bounds=bounds)
        # res.fun = optimal t (minimized maximum violation)
        return (1 - lamb) * np.max(np.maximum(cval, 0)) + lamb * max(0, res.fun)
    except:
        # Fallback if LP fails (e.g., infeasible)
        return (1 - lamb) * np.max(np.maximum(cval, 0)) + lamb * max(0, rho)


def compute_kappa(cval, cgrad, lamb, rho,mc ,n):
    term1 = (1 - lamb) * np.maximum(cval, 0).max()
    obj = np.zeros(n + 1)
    obj[0] = 1.0
    A_ub = np.hstack([-np.ones((mc, 1)), cgrad])
    b_ub = -cval
    bounds = [(0, None)] + [(-rho, rho) for _ in range(n)]

    try:
        res = linprog(c=obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            term2 = lamb * res.fun
        else:
            term2 = lamb * rho
    except:
        term2 = lamb * rho

    return term1 + term2


def compute_kappa_cvxpy(cval, cgrad, lamb, rho,mc ,n):
    first_term = np.maximum(cval, 0).max()

    # Second term: λ * min_d { max_i [ (C_i + ∇C_i^T d)_+ ], ||d||_inf <= ρ }
    d = cp.Variable(n)
    constraints = [cp.norm_inf(d) <= rho]
    obj_terms = [cp.pos(cval[i] + cgrad[i] @ d) for i in range(mc)]
    problem = cp.Problem(cp.Minimize(cp.maximum(*obj_terms)), constraints)
    second_term = problem.solve(solver='PDLP')

    # Combine terms
    kappa = (1 - lamb) * first_term + lamb * second_term
    return kappa

def solvesubp(fgrad, cval, cgrad, kap_val, beta, tau, hesstype, mc, n, qp_solver='osqp', solver_params={}):
    if hesstype == 'diag':
       # P = tau*nx.eye(n)
       P = tau*sp.sparse.identity(n, format='csc')
       kap = kap_val * np.ones(mc)
       cval = np.array(cval)
    return solve_qp(
      P,
      fgrad.reshape((n,)),
      cgrad.reshape((mc, n)),
      kap-cval,
      np.zeros((0, n)),
      np.zeros((0,)),
      -beta*np.ones((n,)),
      beta*np.ones((n,)),
      qp_solver)

def one_sided_loss_constr(loss, net, c_data):
    w_inputs, w_labels = c_data[0]
    b_inputs, b_labels = c_data[1]
    w_outs = net(w_inputs)
    if w_labels.ndim == 0:
        w_labels = w_labels.reshape(1)
        b_labels = b_labels.reshape(1)
    else:
        w_labels = w_labels.unsqueeze(1)
        b_labels = b_labels.unsqueeze(1)
    w_loss = loss(w_outs, w_labels)
    b_outs = net(b_inputs)
    b_loss = loss(b_outs, b_labels)

    return w_loss - b_loss


def StochasticGhost(net, data, w_ind, b_ind, geomp, loss_bound, maxiter, max_runtime=np.inf,
                    stepsize_rule = 'inv_iter', zeta=0.5, gamma0 = 0.1, rho=0.8, lamb=0.5, beta=10., tau=1.,seed=42):
    
    
    loss_fn = torch.nn.BCEWithLogitsLoss()
    
    c1 = lambda net, d: one_sided_loss_constr(loss_fn, net, d) - loss_bound
    c2 = lambda net, d: -one_sided_loss_constr(loss_fn, net, d) - loss_bound
    
    max_sample_size = max([len(w_ind), len(b_ind)])
    
    data_w = torch.utils.data.Subset(data, w_ind)
    data_b = torch.utils.data.Subset(data, b_ind)
    
    history = {'loss': [],
               'constr': [],
               'w': [], 
               'time': [],
               'n_samples': [],
               'fgrad_norm': [], 
               'cgrad_norm': [], 
               'loss_after': [], 
               'constr_after': []}
    
    n = sum(p.numel() for p in net.parameters())
    
    rng = np.random.default_rng(seed=seed)
    
    run_start = timeit.default_timer()
    
    for iteration in range(0, maxiter):
    
        current_time = timeit.default_timer()
        history['time'].append(current_time - run_start)
        
        if max_runtime > 0 and current_time - run_start >= max_runtime:
            print(current_time - run_start)
            history['constr'] = pd.DataFrame(history['constr'])
            return history
    
        # gamma = 
        if stepsize_rule == 'inv_iter':
            gamma = gamma0/(iteration+1)**zeta
        elif stepsize_rule == 'dimin':
            if iteration == 0:
                gamma = gamma0
            else:
                gamma *= (1-zeta*gamma)
        # gamma = 1.5e-2
        
        Nsamp = rng.geometric(p=geomp) - 1
        while (2**(Nsamp+1)) > max_sample_size:
            Nsamp = rng.geometric(p=geomp) - 1
    
        mbatches = [1, 2**(Nsamp+1)]
        history['n_samples'].append(3*(1 + 2 ** (Nsamp+1)))
        dsols = np.zeros((4, n))
        
        ################
        ### sampling ###
        ################
        indices_f, indices_c_w, indices_c_b = [],[],[]
        for j, subp_batch_size in enumerate(mbatches):
            idx_f = rng.choice(len(data), size=subp_batch_size)
            idx_c_w = rng.choice(len(data_w), size=max(subp_batch_size//2, 1))
            idx_c_b = rng.choice(len(data_b), size=max(subp_batch_size//2, 1))
            if j == 1:
                indices_f.append(idx_f[::2]) # even
                indices_f.append(idx_f[1::2]) # odd
                indices_f.append(idx_f) # all
                indices_c_w.append(idx_c_w[::2]) # even
                indices_c_w.append(idx_c_w[1::2]) # odd
                indices_c_w.append(idx_c_w) # all
                indices_c_b.append(idx_c_b[::2]) # even
                indices_c_b.append(idx_c_b[1::2]) # odd
                indices_c_b.append(idx_c_b) # all
            else:
                indices_f.append(idx_f)
                indices_c_w.append(idx_c_w)
                indices_c_b.append(idx_c_b)
        ##############
        ### update ###
        ##############
        for j, samples in enumerate(zip(indices_f, indices_c_w, indices_c_b)):
            net.zero_grad()
            
            idx = samples[0]
            idx_w = samples[1]
            idx_b = samples[2]
            obj_batch = data[idx]
            c1_batch = [data_w[idx_w], data_b[idx_b]]
            c2_batch = [data_w[idx_w], data_b[idx_b]]
            
            # calculate autograd jacobian of obj fun w.r.t. params
            outs = net(obj_batch[0])
            feval = loss_fn(outs, obj_batch[1].unsqueeze(1))
    
            feval.backward()
            dfdw = net_grads_to_tensor(net, clip=False)
            net.zero_grad()
            
            # calculate autograd jacobian of constraints fun w.r.t. params
            c1_val = c1(net, c1_batch)
            c1_val.backward()
            c1_grad = ar.to_numpy(net_grads_to_tensor(net, clip=False))
            
            net.zero_grad()
            c2_val =  c2(net, c2_batch)
            c2_val.backward()
            c2_grad = ar.to_numpy(net_grads_to_tensor(net,clip=False))
            
            constraint_eval = np.array([c1_val.detach(), c2_val.detach()])
            dcdw = np.array([c1_grad, c2_grad])
            
            history['constr'].append(np.array([c1_val.detach().numpy(), c2_val.detach().numpy()]))
            
            # kappa = compute_kappa_cvxpy(constraint_eval, dcdw, rho, lamb, mc=2, n=len(dfdw))
            kappa = compute_kappa(constraint_eval, dcdw, rho, lamb, mc=2, n=len(dfdw))
            # kappa = computekappa(constraint_eval, dcdw, rho, lamb, mc=2, n=len(dfdw), scalef=1)
            # kappa = __computekappa__(constraint_eval, dcdw, rho, lamb, mc=2, n=len(dfdw))
            
            # solve subproblem
            feval = ar.to_numpy(feval)
            dfdw = ar.to_numpy(dfdw)
            dsol = solvesubp(dfdw,
                                constraint_eval, dcdw,
                                kappa, beta, tau,
                                hesstype='diag', mc=2, n=len(dfdw),
                                qp_solver='osqp')
            
            dsols[j, :] = dsol
        
        # aggregate solutions to the subproblem according to Eq. 23
        dsol = dsols[0, :] + (dsols[3, :]-0.5*dsols[1, :] -
                            0.5*dsols[2, :])/(geomp*((1-geomp)**(Nsamp)))
        
        start = 0
        print(f'{iteration}', end='\r')
        with torch.no_grad():
            w = net_params_to_tensor(net)
            if any([torch.any(torch.isnan(lw)) for lw in w]):
                print('NaNs!')
                return history
            for i in range(len(w)):
                end = start + w[i].numel()
                w[i].add_(torch.tensor(gamma*np.reshape(dsol[start:end], np.shape(w[i]))))
                start = end
                
        history['w'].append(deepcopy(net.state_dict()))
        
        feval = loss_fn(outs, obj_batch[1].unsqueeze(1))
    
    history['constr'] = pd.DataFrame(history['constr'])
    return history