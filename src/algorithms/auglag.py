from typing import Tuple,Callable, List
from matplotlib import pyplot as plt
import torch.utils.data.dataloader
import numpy as np
import scipy as sp
from copy import deepcopy
from scipy.optimize import linprog
from qpsolvers import solve_qp
import autoray as ar
import timeit
from itertools import cycle

from src.algorithms.c_utils.constraint_fns import *
from .utils import net_grads_to_tensor, net_params_to_tensor
import cvxpy as cp
import torch
from fairret.statistic import *
from fairret.loss import NormLoss

m_det = 0
m_st = 2
import timeit
constr_sampling_interval = 1
max_runtime = 15


def AugLagr(net: torch.nn.Module, dataset, w_ind, b_ind, batch_size, loss_bound, maxiter, max_runtime=np.inf,
                lambda_bound = 1e1,
                pmult_bound = 1e1,
                start_lambda=None,
                update_lambda=True,
                update_pen = True,
                device='cpu',
                seed=None,
                epochs=1,
                c_sample_half=False):
        
    history = {'loss': [],
               'constr': [],
               'w': [],
               'time': [],
               'n_samples': []}
    
     
    c1 = lambda net, d: one_sided_loss_constr(loss_fn, net, d) - loss_bound
    c2 = lambda net, d: -one_sided_loss_constr(loss_fn, net, d) - loss_bound
    
    # statistic = PositiveRate()
    # norm_fairret = NormLoss(statistic)
    # c1 = lambda net, d: fairret_pr_constr(norm_fairret, net, d) - loss_bound
    # c2 = lambda net, d: fairret_pr_constr(norm_fairret, net, d) - loss_bound
    data_w = torch.utils.data.Subset(dataset, w_ind)
    data_b = torch.utils.data.Subset(dataset, b_ind)
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)
    c_bs = max(batch_size//2, 1) if c_sample_half else batch_size
    loader_w = cycle(torch.utils.data.DataLoader(data_w, c_bs, shuffle=True, generator=gen))
    loader_b = cycle(torch.utils.data.DataLoader(data_b, c_bs, shuffle=True, generator=gen))
    
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-3)
    n = sum(p.numel() for p in net.parameters())
    
    _lambda = torch.zeros(2) if start_lambda is None else start_lambda
    if update_pen:
        c = 1.
    else:
        c = 10
    l_stepsize = 1e-1
    p = 2.
    
    run_start = timeit.default_timer()
    for epoch in range(epochs):
        loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, generator=gen)
        for iteration, data in enumerate(loader):
            net.zero_grad()
            
            current_time = timeit.default_timer()
            history['time'].append(current_time - run_start)
            history['w'].append(deepcopy(net.state_dict()))
            history['n_samples'].append(batch_size*3)
            if max_runtime > 0 and current_time - run_start >= max_runtime:
                print(current_time - run_start)
                return history
            
            w_sample = next(loader_w)
            b_sample = next(loader_b)
            
            inputs, labels = data
            outputs = net(inputs)
            if labels.dim() < outputs.dim():
                labels = labels.unsqueeze(1)
            loss_eval = loss_fn(outputs, labels)
            # constraint_eval = torch.tensor([
            #     c1(net, [w_sample, b_sample]), 
            #     c2(net, [w_sample, b_sample])
            # ])
            constraint_eval = torch.concat([
                c1(net, [w_sample, b_sample]).unsqueeze(0), 
                c2(net, [w_sample, b_sample]).unsqueeze(0)
            ])
            constraint_eval = torch.maximum(constraint_eval, torch.zeros(2))
            history['constr'].append(constraint_eval.cpu().detach().numpy())
            history['loss'].append(loss_eval.cpu().detach().numpy())
            
            penalty_term = (1/p) * torch.sum(c * torch.abs(constraint_eval)**p)  # torch.norm(constraint_eval, p=_p)
            lag_term = _lambda @ constraint_eval
            L = loss_eval + lag_term + penalty_term
        
            L.backward()
            optimizer.step()
            
            with np.printoptions(precision=3, suppress=True, floatmode='fixed'):
                print(f'{iteration:3}|{loss_eval.detach().cpu().numpy():5}|{_lambda.detach().cpu().numpy()}|{constraint_eval}|{c}', end='\r')
            # print(f'{iteration}', end='\r')
            # recalculate objective and constraint values based on updated network weights
            
            constraint_eval_updated = torch.tensor([c1(net, [w_sample, b_sample]), c2(net, [w_sample, b_sample])])
            constraint_eval_updated = torch.maximum(constraint_eval_updated, torch.zeros(2))
            # with torch.inference_mode():
            for i in range(len(constraint_eval_updated)):
                if iteration != 0 and constraint_eval_updated[i] > 0.5*constraint_eval[i]:
                    lambda_upd = _lambda[i] + (3e-2) * c * torch.abs(constraint_eval_updated[i])**(p-1)
                    # if update_lambda and lambda_upd < lambda_bound:
                        # _lambda[i] = lambda_upd
                    if lambda_upd > lambda_bound:
                        lambda_upd = 0.
                    _lambda[i] = lambda_upd

            # if c*1.01 < pmult_bound:
            #     c *= 1.01
                
                
        # old_constraint_eval = deepcopy(constraint_eval)
    
    # plt.plot(history['loss'])
    # plt.show()
    # plt.plot(history['constr'])
    # plt.show()
    return history


def AugLagr_unbiased(net: torch.nn.Module, data, w_ind, b_ind, batch_size, loss_bound, maxiter, max_runtime=np.inf,
                lambda_bound = 1e6,
                pmult_bound = 1e6,
                start_lambda=None,
                update_lambda=True,
                update_pen = True,
                device='cpu',
                seed=None,
                c_sample_half=True):
        
    history = {'loss': [],
               'constr': [],
               'w': [],
               'time': [],
               'n_samples': []}
    
    c1 = lambda net, d: torch.maximum(one_sided_loss_constr(loss_fn, net, d) - loss_bound, torch.tensor(0.))
    c2 = lambda net, d: torch.maximum(-one_sided_loss_constr(loss_fn, net, d) - loss_bound, torch.tensor(0.))
    
    c = [c1, c2]
    data_w = torch.utils.data.Subset(data, w_ind)
    data_b = torch.utils.data.Subset(data, b_ind)
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)
    loader = torch.utils.data.DataLoader(data, batch_size, shuffle=True, generator=gen)
    c_bs = max(batch_size//2, 1) if c_sample_half else batch_size
    loader_w = cycle(torch.utils.data.DataLoader(data_w, c_bs, shuffle=True, generator=gen))
    loader_b = cycle(torch.utils.data.DataLoader(data_b, c_bs, shuffle=True, generator=gen))
    
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(net.parameters())
    n = sum(p.numel() for p in net.parameters())
    
    _lambda = torch.zeros(2) if start_lambda is None else start_lambda
    pm = 10
    beta = 2.
    p = 2.
    
    run_start = timeit.default_timer()
    for iteration, data in enumerate(loader):
        net.zero_grad()
        
        current_time = timeit.default_timer()
        history['time'].append(current_time - run_start)
        history['w'].append(deepcopy(net.state_dict()))
        history['n_samples'].append(batch_size*3)
        if max_runtime > 0 and current_time - run_start >= max_runtime:
            print(current_time - run_start)
            return history
        
        # objective and obj grad estimate
        
        f_inputs, f_labels = data
        outputs = net(f_inputs)
        if f_labels.dim() < outputs.dim():
            f_labels = f_labels.unsqueeze(1)
        loss_eval = loss_fn(outputs, f_labels)
        loss_eval.backward() # loss grad
        f_grad = net_grads_to_tensor(net)
        net.zero_grad()
        
        # constraint grad estimate 1 
        
        cw_sample = next(loader_w)
        cb_sample = next(loader_b)
        c_sample = [cw_sample, cb_sample]
        c_1 = [ci(net, c_sample).reshape(1) for ci in c]
        
        c_grad = []
        for ci in c_1:
            ci.backward()
            ci_grad = net_grads_to_tensor(net)
            c_grad.append(ci_grad)
            net.zero_grad()
        c_grad = torch.stack(c_grad)
        
        # constraint value estimate
        
        cw_sample = next(loader_w)
        cb_sample = next(loader_b)
        c_sample = [cw_sample, cb_sample]
        with torch.no_grad():
            c_2 = torch.concat([ci(net, c_sample).reshape(1) for i, ci in enumerate(c)])
            
        penalty_term_grad = pm*(c_grad.T @ c_2)
        lag_term_grad = c_grad.T @ _lambda
        G = f_grad + lag_term_grad + penalty_term_grad
        
        # manually set gradients
        end = 0
        for layer in net.parameters():
            _grad = G[end:end+np.prod(layer.shape)].reshape(layer.shape)
            layer.grad = _grad
            end += np.prod(layer.shape)
        
        # x_t = torch.concat(net_params_to_tensor(net, flatten=True, copy=True))
        
        # penalty_term = (1/p) * torch.sum(pm * torch.abs(constraint_eval)**p)  # torch.norm(constraint_eval, p=_p)
        # lag_term = _lambda @ constraint_eval
        # L = loss_eval + lag_term + penalty_term
    
        # L.backward()
        optimizer.step()        
        
        with np.printoptions(precision=6, suppress=True, floatmode='fixed'):
            print(f'{iteration:5}|{loss_eval.detach().cpu().numpy()}|{_lambda.detach().cpu().numpy()}|{c_2.cpu().numpy()}|{pm}', end='\r')
        # print(f'{iteration}', end='\r')        
        # recalculate objective and constraint values based on updated network weights
        
        constraint_eval_updated = torch.tensor([c1(net, [cw_sample, cb_sample]), c2(net, [cw_sample, cb_sample])])
        constraint_eval_updated = torch.maximum(constraint_eval_updated, torch.zeros(2))
        # with torch.inference_mode():
        for i in range(len(constraint_eval_updated)):
            if iteration != 0 and constraint_eval_updated[i] > 0.5*c_1[i]:
                lambda_upd = _lambda[i] + pm * torch.abs(constraint_eval_updated[i])**(p-1)
                if update_lambda and lambda_upd < lambda_bound:
                    _lambda[i] = lambda_upd
            
        if pm < pmult_bound:
            pm *= 2 
                
                
        # old_constraint_eval = deepcopy(constraint_eval)
        
    return history