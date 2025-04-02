from typing import Tuple,Callable, List
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
       #print("IMPORTANT!!!!!",res.fun)
       return (1-lamb)*max_c_viol + lamb*max(0, res.fun)
    #    return (1-lamb)*max(0, sum(cval)) + lamb*max(0, res.fun)
    except:
       return (1-lamb)*max_c_viol + lamb*max(0, rho)


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
                    zeta=0.7, gamma0 = 0.1, rho=1e-3, lamb=0.5, beta=10., tau=2., random_state=42):
    
    
    loss_fn = torch.nn.BCEWithLogitsLoss()
    
    c1 = lambda net, d: one_sided_loss_constr(loss_fn, net, d) - loss_bound
    c2 = lambda net, d: -one_sided_loss_constr(loss_fn, net, d) - loss_bound
    
    max_sample_size = max([len(w_ind), len(b_ind)])
    
    data_w = torch.utils.data.Subset(data, w_ind)
    data_b = torch.utils.data.Subset(data, b_ind)
    
    history = {'loss': [],
               'constr': [],
               'w': [], 
               'fgrad_norm': [], 
               'cgrad_norm': [], 
               'loss_after': [], 
               'constr_after': []}

    if isinstance(data, torch.utils.data.Dataset):
        obj_dataloader = torch.utils.data.DataLoader(data,batch_size=1)
    
    n = sum(p.numel() for p in net.parameters())
    
    rng = np.random.default_rng(seed=random_state)
    
    run_start = timeit.default_timer()
    
    for iteration in range(0, maxiter):
    
        current_time = timeit.default_timer()
        
        if max_runtime > 0 and current_time - run_start >= max_runtime:
            print(current_time - run_start)
            return
    
        gamma = gamma0/(iteration+1)**zeta
        
        
        Nsamp = rng.geometric(p=geomp)
        while (2**(Nsamp+1)) > max_sample_size:
            Nsamp = rng.geometric(p=geomp)
    
        mbatches = [1, 2**Nsamp, 2**Nsamp, 2**(Nsamp+1)]
        dsols = np.zeros((4, n))
        # for each subproblem:        
        for j, subp_batch_size in enumerate(mbatches):
            idx = rng.choice(len(data), size=subp_batch_size)
            obj_batch = data[idx]
            
            # calculate autograd jacobian of obj fun w.r.t. params
            net.zero_grad()
            outs = net(obj_batch[0])
            feval = loss_fn(outs, obj_batch[1].unsqueeze(1))
    
            feval.backward()
            dfdw = net_grads_to_tensor(net, clip=False)
            
            idx_w = rng.choice(len(data_w), size=subp_batch_size)
            idx_b = rng.choice(len(data_b), size=subp_batch_size)
            c1_sample = [data_w[idx_w], data_b[idx_b]]
            c2_sample = [data_w[idx_w], data_b[idx_b]]
            
            # calculate autograd jacobian of constraints fun w.r.t. params
            net.zero_grad()
            c1_val = c1(net, c1_sample)
            c1_val.backward()
            c1_grad = ar.to_numpy(net_grads_to_tensor(net, clip=False))
            
            net.zero_grad()
            c2_val =  c2(net, c2_sample)
            c2_val.backward()
            c2_grad = ar.to_numpy(net_grads_to_tensor(net,clip=False))
            
            constraint_eval = np.array([c1_val.detach(), c2_val.detach()])
            dcdw = np.array([c1_grad, c2_grad])
            
            kappa = computekappa(constraint_eval, dcdw, rho, lamb, mc=2, n=len(dfdw), scalef=1)
            
            # solve subproblem
            feval = ar.to_numpy(feval)
            dfdw = ar.to_numpy(dfdw)
            dsol = solvesubp(dfdw,
                                constraint_eval, dcdw,
                                kappa, beta, tau,
                                hesstype='diag', mc=2, n=len(dfdw),
                                qp_solver='osqp')
            
            dsols[j, :] = dsol
    
            history['loss'].append(feval)
            history['constr'].append(constraint_eval)
            history['fgrad_norm'].append(np.linalg.norm(dfdw))
            history['cgrad_norm'].append(np.linalg.norm(dcdw))
        
        # aggregate solutions to the subproblem according to Eq. 23
        dsol = dsols[0, :] + (dsols[3, :]-0.5*dsols[1, :] -
                            0.5*dsols[2, :])/(geomp*((1-geomp)**Nsamp))
        
        start = 0
        print(f'{iteration}', end='\r')
        with torch.no_grad():
            w = net_params_to_tensor(net)
            for i in range(len(w)):
                end = start + w[i].numel()
                w[i].add_(torch.tensor(gamma*np.reshape(dsol[start:end], np.shape(w[i]))))
                start = end
                
        history['w'].append(deepcopy(net.state_dict()))
        
    return history
    


def StochasticGhost_OddEven(net, data, w_ind, b_ind, geomp, loss_bound, maxiter, max_runtime=np.inf,
                    zeta=0.7, gamma0 = 0.1, rho=1e-3, lamb=0.5, beta=10., tau=2.,random_state=42):
    
    
    loss_fn = torch.nn.BCEWithLogitsLoss()
    
    c1 = lambda net, d: one_sided_loss_constr(loss_fn, net, d) - loss_bound
    c2 = lambda net, d: -one_sided_loss_constr(loss_fn, net, d) - loss_bound
    
    max_sample_size = max([len(w_ind), len(b_ind)])
    
    data_w = torch.utils.data.Subset(data, w_ind)
    data_b = torch.utils.data.Subset(data, b_ind)
    
    history = {'loss': [],
               'constr': [],
               'w': [], 
               'n_samples': [],
               'fgrad_norm': [], 
               'cgrad_norm': [], 
               'loss_after': [], 
               'constr_after': []}
    
    n = sum(p.numel() for p in net.parameters())
    
    rng = np.random.default_rng(seed=random_state)
    
    run_start = timeit.default_timer()
    
    for iteration in range(0, maxiter):
    
        current_time = timeit.default_timer()
        
        if max_runtime > 0 and current_time - run_start >= max_runtime:
            print(current_time - run_start)
            return
    
        gamma = gamma0/(iteration+1)**zeta
        
        
        Nsamp = rng.geometric(p=geomp)
        while (2**(Nsamp+1)) > max_sample_size:
            Nsamp = rng.geometric(p=geomp)
    
        mbatches = [1, 2**(Nsamp+1)]
        history['n_samples'].append(3*(1 + 2 ** (Nsamp+1)))
        dsols = np.zeros((4, n))
        # for each subproblem:
        # same samples for each constraint? or no?
        
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
            
            # calculate autograd jacobian of constraints fun w.r.t. params
            net.zero_grad()
            c1_val = c1(net, c1_batch)
            c1_val.backward()
            c1_grad = ar.to_numpy(net_grads_to_tensor(net, clip=False))
            
            net.zero_grad()
            c2_val =  c2(net, c2_batch)
            c2_val.backward()
            c2_grad = ar.to_numpy(net_grads_to_tensor(net,clip=False))
            
            constraint_eval = np.array([c1_val.detach(), c2_val.detach()])
            dcdw = np.array([c1_grad, c2_grad])
            
            kappa = computekappa(constraint_eval, dcdw, rho, lamb, mc=2, n=len(dfdw), scalef=1)
            
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
                            0.5*dsols[2, :])/(geomp*((1-geomp)**Nsamp))
        
        start = 0
        print(f'{iteration}', end='\r')
        with torch.no_grad():
            w = net_params_to_tensor(net)
            for i in range(len(w)):
                end = start + w[i].numel()
                w[i].add_(torch.tensor(gamma*np.reshape(dsol[start:end], np.shape(w[i]))))
                start = end
                
        history['w'].append(deepcopy(net.state_dict()))
        
        feval = loss_fn(outs, obj_batch[1].unsqueeze(1))
        # history['loss'].append(feval)
        # history['constr'].append(constraint_eval)
        
    return history