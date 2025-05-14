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

from src.algorithms.c_utils.constraint_fns import *
from .utils import net_grads_to_tensor, net_params_to_tensor
import cvxpy as cp
import torch

m_det = 0
m_st = 2
import timeit
constr_sampling_interval = 1

def project(x, m):
    # for i in range(1,m+1):
    #     if x[-i] < 0:
    #         x[-i] = 0
    return x










# def SwitchingSubgradient(net: torch.nn.Module, data, w_ind, b_ind, loss_bound,
#             batch_size=8,
#             max_runtime=np.inf,
#             # ctol_rule = 'dimin',
#             ctol = 1e-1,
#             # ctol_min = 1e-5,
#             f_stepsize_rule = 'dimin',
#             f_stepsize = 5e-1,
#             c_stepsize_rule = 'adaptive',
#             c_stepsize = None,
#             device='cpu',
#             epochs=1,
#             seed = 42):
        
#     history = {'loss': [],
#                'constr': [],
#                'w': [],
#                'time': []}
    
#     c1 = lambda net, d: one_sided_loss_constr(loss_fn, net, d) - loss_bound
#     c2 = lambda net, d: -one_sided_loss_constr(loss_fn, net, d) - loss_bound
    
#     data_w = torch.utils.data.Subset(data, w_ind)
#     data_b = torch.utils.data.Subset(data, b_ind)
    
#     loss_fn = torch.nn.BCEWithLogitsLoss()
#     loss_eval = None
#     c_t = None
#     run_start = timeit.default_timer()
#     current_time = timeit.default_timer()
    
#     f_eta_t = f_stepsize
#     c_eta_t = c_stepsize
    
#     f_iters = 0
#     c_iters = 0
#     for epoch in range(epochs):
        
#         if current_time - run_start >= max_runtime:
#             break
        
#         gen = torch.Generator(device=device)
#         gen.manual_seed(seed+epoch)
#         loader = torch.utils.data.DataLoader(data, batch_size, shuffle=True, generator=gen)
#         loader_w = cycle(torch.utils.data.DataLoader(data_w, batch_size, shuffle=True, generator=gen))
#         loader_b = cycle(torch.utils.data.DataLoader(data_b, batch_size, shuffle=True, generator=gen))
        
#         for iteration, f_sample in enumerate(loader):
            
#             current_time = timeit.default_timer()
#             history['time'].append(current_time - run_start)
#             if max_runtime > 0 and current_time - run_start >= max_runtime:
#                 print(current_time - run_start)
#                 break
            
#             net.zero_grad()
                
#             # if ctol_rule == 'dimin':
#             #     ctol_t = ctol/np.sqrt(iteration+1)
#             #     if ctol_t < ctol_min:
#             #         ctol_t = ctol_min
#             # elif ctol_rule == 'const':
#             #     ctol_t = ctol
            
#             # generate sample of constraints
#             cw_sample = next(loader_w)
#             cb_sample = next(loader_b)
#             c_sample = [cw_sample, cb_sample]
#             c1_eval = c1(net, c_sample)
#             c2_eval = c2(net, c_sample)
#             c_t = torch.concat([
#                 c1_eval.reshape(1),
#                 c2_eval.reshape(1)
#             ])
#             c_max = torch.max(c_t)
#             history['constr'].append(c_max.cpu().detach().numpy())

#             x_t = net_params_to_tensor(net, flatten=True, copy=True)
            
#             if c_max >= ctol:
#                 c_iters += 1
#                 c_max.backward()
#                 c_grad = net_grads_to_tensor(net)
#                 if c_stepsize_rule == 'adaptive':
#                     c_eta_t = c_max / torch.norm(c_grad)**2
#                 elif c_stepsize_rule == 'const':
#                     c_eta_t = c_stepsize
#                 elif c_stepsize_rule == 'dimin':
#                     c_eta_t = c_stepsize / np.sqrt(c_iters)
                
#                 x_t1 = project(x_t - c_eta_t*c_grad, m=2)
#             else:
#                 f_iters += 1
#                 f_inputs, f_labels = f_sample
#                 outputs = net(f_inputs)
#                 if f_labels.dim() < outputs.dim():
#                     f_labels = f_labels.unsqueeze(1)
#                 loss_eval = loss_fn(outputs, f_labels)
#                 loss_eval.backward()
#                 history['loss'].append(loss_eval.cpu().detach().numpy())
#                 f_grad = net_grads_to_tensor(net)
                
#                 if f_stepsize_rule == 'dimin':
#                     f_eta_t = f_stepsize / np.sqrt(f_iters)
#                 elif f_stepsize_rule == 'const':
#                     f_eta_t = f_stepsize
#                 x_t1 = project(x_t - f_eta_t*f_grad, m=2)
            
#             start = 0
#             with torch.no_grad():
#                 w = net_params_to_tensor(net, flatten=False, copy=False)
#                 for i in range(len(w)):
#                     end = start + w[i].numel()
#                     w[i].set_(x_t1[start:end].reshape(w[i].shape))
#                     start = end
            
#             if loss_eval is not None and c_t is not None:
#                 with np.printoptions(precision=6, suppress=True):
#                     print(f'{iteration:5}|{loss_eval.detach().cpu().numpy()}|{c_t.detach().cpu().numpy()}', end='\r')
#             history['w'].append(deepcopy(net.state_dict()))
        
#     ######################
#     ### POSTPROCESSING ###    
#     ######################
#     print('\n')
#     print(c_iters)
#     return history



def SwitchingSubgradient_unbiased(net: torch.nn.Module, data, w_ind, b_ind, loss_bound,
            # ctol_rule = 'dimin',
            ctol,
            # ctol_min = 1e-5,
            f_stepsize_rule,
            f_stepsize,
            c_stepsize_rule,
            c_stepsize,
            device='cpu',
            batch_size=8,
            epochs=1,
            seed = 42,
            max_runtime=np.inf):
        
    history = {'loss': [],
               'constr': [],
               'w': [],
               'time': [],
               'n_samples': []}
    
    c1 = lambda net, d: one_sided_loss_constr(loss_fn, net, d) - loss_bound
    c2 = lambda net, d: -one_sided_loss_constr(loss_fn, net, d) - loss_bound
    
    c = [c1, c2]
    
    data_w = torch.utils.data.Subset(data, w_ind)
    data_b = torch.utils.data.Subset(data, b_ind)
    
    loss_fn = torch.nn.BCEWithLogitsLoss()
    loss_eval = None
    c_t = None
    run_start = timeit.default_timer()
    current_time = timeit.default_timer()
    
    f_eta_t = f_stepsize
    c_eta_t = c_stepsize
    
    f_iters = 0
    c_iters = 0
    for epoch in range(epochs):
        
        if current_time - run_start >= max_runtime:
            break
        gen = torch.Generator(device=device)
        gen.manual_seed(seed+epoch)
        loader = torch.utils.data.DataLoader(data, batch_size, shuffle=True, generator=gen)
        loader_w = cycle(torch.utils.data.DataLoader(data_w, batch_size, shuffle=True, generator=gen))
        loader_b = cycle(torch.utils.data.DataLoader(data_b, batch_size, shuffle=True, generator=gen))
        
        for iteration, f_sample in enumerate(loader):
            current_time = timeit.default_timer()
            history['time'].append(current_time - run_start)
            if max_runtime > 0 and current_time - run_start >= max_runtime:
                print(current_time - run_start)
                break
            
            net.zero_grad()
            
            # generate sample of constraints
            cw_sample = next(loader_w)
            cb_sample = next(loader_b)
            c_sample = [cw_sample, cb_sample]
            # calc constraints and update multipliers (line 3)
            with torch.no_grad():
                c_t = torch.cat([ci(net, c_sample).reshape(1) for i, ci in enumerate(c)])
                c_max = torch.max(c_t)
            history['constr'].append(c_max.cpu().detach().numpy())

            x_t = net_params_to_tensor(net, flatten=True, copy=True)
            
            if c_max >= ctol:
                c_iters += 1
            
                history['n_samples'].append(batch_size*2)
                # calculate grad on an independent sample
                cw_sample = next(loader_w)
                cb_sample = next(loader_b)
                c_sample = [cw_sample, cb_sample]
                c_t2 = torch.concat([ci(net, c_sample).reshape(1) for i, ci in enumerate(c)])
                c_max2 = torch.max(c_t2)
                c_max2.backward()
                c_grad = net_grads_to_tensor(net)
                if c_stepsize_rule == 'adaptive':
                    c_eta_t = c_max / torch.norm(c_grad)**2
                elif c_stepsize_rule == 'const':
                    c_eta_t = c_stepsize
                elif c_stepsize_rule == 'dimin':
                    c_eta_t = c_stepsize / np.sqrt(c_iters)
                
                x_t1 = project(x_t - c_eta_t*c_grad, m=2)
            else:
            
                history['n_samples'].append(batch_size)
                f_iters += 1
                f_inputs, f_labels = f_sample
                outputs = net(f_inputs)
                if f_labels.dim() < outputs.dim():
                    f_labels = f_labels.unsqueeze(1)
                loss_eval = loss_fn(outputs, f_labels)
                loss_eval.backward()
                history['loss'].append(loss_eval.cpu().detach().numpy())
                f_grad = net_grads_to_tensor(net)
                
                if f_stepsize_rule == 'dimin':
                    f_eta_t = f_stepsize / np.sqrt(f_iters)
                elif f_stepsize_rule == 'const':
                    f_eta_t = f_stepsize
                x_t1 = project(x_t - f_eta_t*f_grad, m=2)
            
            start = 0
            with torch.no_grad():
                w = net_params_to_tensor(net, flatten=False, copy=False)
                for i in range(len(w)):
                    end = start + w[i].numel()
                    w[i].set_(x_t1[start:end].reshape(w[i].shape))
                    start = end
            
            if loss_eval is not None and c_t is not None:
                with np.printoptions(precision=6, suppress=True):
                    print(f'{epoch:2} | {iteration:5}|{loss_eval.detach().cpu().numpy()}|{c_t.detach().cpu().numpy()}', end='\r')
            history['w'].append(deepcopy(net.state_dict()))
        
    ######################
    ### POSTPROCESSING ###    
    ######################
    
    
    
    print('\n')
    print(c_iters)
    return history