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

m_det = 0
m_st = 2
import timeit
constr_sampling_interval = 1
max_runtime = 15

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


def project(x, m):
    for i in range(1,m+1):
        if x[-i] < 0:
            x[-i] = 0
    return x


def SSLALM(net: torch.nn.Module, data, w_ind, b_ind, loss_bound,
            batch_size=1,
            max_runtime=np.inf,
            lambda_bound = 100,
            pmult_bound = 1e3,
            nu = 1e-4,
            mu = 3.,
            tau = 1e-3,
            beta = 0.1,
            eta = 1e-3,
            start_lambda=None,
            update_lambda=True,
            update_pen = True,
            device='cpu'):
        
    history = {'loss': [],
               'constr': [],
               'w': []}
    
    # slack variables
    slack_vars = torch.zeros(2, requires_grad=True)
    
    c1 = lambda net, d, s: one_sided_loss_constr(loss_fn, net, d) - loss_bound + s
    c2 = lambda net, d, s: -one_sided_loss_constr(loss_fn, net, d) - loss_bound + s
    
    data_w = torch.utils.data.Subset(data, w_ind)
    data_b = torch.utils.data.Subset(data, b_ind)    
    loader = torch.utils.data.DataLoader(data, batch_size, shuffle=True, generator=torch.Generator(device=device))
    loader_w = cycle(torch.utils.data.DataLoader(data_w, batch_size, shuffle=True, generator=torch.Generator(device=device)))
    loader_b = cycle(torch.utils.data.DataLoader(data_b, batch_size, shuffle=True, generator=torch.Generator(device=device)))
    
    loss_fn = torch.nn.BCEWithLogitsLoss()
    n = sum(p.numel() for p in net.parameters())
    
    _lambda = torch.zeros(2,requires_grad=True) if start_lambda is None else start_lambda
    rho = 5
    z = torch.concat([
            net_params_to_tensor(net, flatten=True, copy=True),
            slack_vars
        ])
    
    run_start = timeit.default_timer()
    for iteration, f_sample in enumerate(loader):
        if iteration == 10000:
            break
        
        net.zero_grad()
        
        current_time = timeit.default_timer()
        if max_runtime > 0 and current_time - run_start >= max_runtime:
            print(current_time - run_start)
            return
        
        ########################
        ## UPDATE MULTIPLIERS ##
        ########################
        
        # sample for constraints
        with torch.no_grad():
            cw_sample = next(loader_w)
            cb_sample = next(loader_b)
            c_sample = [cw_sample, cb_sample]
            c_t = torch.tensor([
                c1(net, c_sample, slack_vars[0]),
                c2(net, c_sample, slack_vars[1])
            ])
            _lambda += eta * c_t
        
        if torch.norm(_lambda) >= lambda_bound:
            _lambda = torch.zeros_like(_lambda, requires_grad=True)
        
        #######################
        ## UPDATE PARAMETERS ##
        #######################
        
        # calculate Lagrangian
        net.zero_grad()
        slack_vars.grad = None
        # loss
        f_inputs, f_labels = f_sample
        outputs = net(f_inputs)       
        if f_labels.dim() < outputs.dim():
            f_labels = f_labels.unsqueeze(1)
        loss_eval = loss_fn(outputs, f_labels)
        # loss grad
        loss_eval.backward()
        f_grad = net_grads_to_tensor(net)
        f_grad = torch.concat([f_grad, torch.zeros(2)]) # add zeros for slack vars
        net.zero_grad()
        
        # constraints
        # cw_sample = next(loader_w)
        # cb_sample = next(loader_b)
        # c_sample = [cw_sample, cb_sample]
        _c1 = c1(net, c_sample, slack_vars[0]).reshape(1)
        _c2 = c2(net, c_sample, slack_vars[1]).reshape(1)
        constraint_eval = torch.concat([_c1,_c2])
        # constraint grads
        _c1.backward()
        c1_grad = net_grads_to_tensor(net)
        c1_grad = torch.concat([c1_grad, slack_vars.grad])
        net.zero_grad()
        slack_vars.grad = None
        
        _c2.backward()
        c2_grad = net_grads_to_tensor(net)
        c2_grad = torch.concat([c2_grad, slack_vars.grad])
        net.zero_grad()
        slack_vars.grad = None
        
        with torch.no_grad():
            cw_sample = next(loader_w)
            cb_sample = next(loader_b)
            c_sample = [cw_sample, cb_sample]
            _c1_ = c1(net, c_sample, slack_vars[0]).reshape(1)
            _c2_ = c2(net, c_sample, slack_vars[1]).reshape(1)
            _c_ = torch.tensor([_c1_, _c2_])
            
        c_grad = torch.stack([c1_grad, c2_grad])
        
        x_t = torch.concat([
            net_params_to_tensor(net, flatten=True, copy=True),
            slack_vars
        ])
        
        G = f_grad + c_grad.T @ _lambda + rho*(c_grad.T @ _c_) + mu*(x_t - z)

        x_t1 = project(x_t - tau*G, 2)
        z += beta*(x_t-z)
        
        start = 0
        with torch.no_grad():
            w = net_params_to_tensor(net, flatten=False, copy=False)
            for i in range(len(w)):
                end = start + w[i].numel()
                w[i].set_(x_t1[start:end].reshape(w[i].shape))
                start = end

            for i in range(len(slack_vars)):
                slack_vars[i] = x_t1[i-len(slack_vars)]
                
        print(f"""{iteration}|{loss_eval.detach().cpu().numpy()}|{_lambda.detach().cpu().numpy()}|{constraint_eval.detach().cpu().numpy()}|{slack_vars.detach().cpu().numpy()}""", end='\r')
        history['w'].append(deepcopy(net.state_dict()))
        
    return history