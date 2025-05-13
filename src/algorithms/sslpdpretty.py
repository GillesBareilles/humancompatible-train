from typing import Tuple,Callable, List, Iterable
import torch.utils.data.dataloader
import numpy as np
import scipy as sp
from copy import deepcopy
from scipy.optimize import linprog
from qpsolvers import solve_qp
import autoray as ar
import timeit
from itertools import cycle

from src.algorithms.constraints import *
from src.algorithms.c_utils.constraint import FairnessConstraint
from fairret.statistic import *
# from fairret.metric import *
from fairret.loss import *
from .utils import *
from .constraints import *
import torch

# m_det = 0
# m_st = 2
import timeit
constr_sampling_interval = 1
max_runtime = 15


def project(x, m):
    for i in range(1,m+1):
        if x[-i] < 0:
            x[-i] = 0
    return x


def SSLPD_new(net: torch.nn.Module,
          data: torch.utils.data.Dataset,
          constraints: Iterable[FairnessConstraint],
          batch_size=1,
          max_runtime=np.inf,
          lambda_bound = 100,
          rho = 5, mu = 2., tau = 1e-4, beta = 0.1, eta = 5e-2, B = 1000, start_lambda=None,
          device='cpu', epochs=1, seed = 42, max_iter = None):
        
    history = {'loss': [],
               'constr': [],
               'w': [],
               'time': [],
               'n_samples': []}
    
    loss_fn = torch.nn.BCEWithLogitsLoss()
    
    m = len(constraints)
    slack_vars = torch.zeros(m, requires_grad=True)
    _lambda = torch.zeros(m, requires_grad=True) if start_lambda is None else start_lambda
    z = torch.concat([
            net_params_to_tensor(net, flatten=True, copy=True),
            slack_vars
        ])
    c = constraints
    run_start = timeit.default_timer()
    for epoch in range(epochs):
        
        gen = torch.Generator(device=device)
        gen.manual_seed(seed+epoch)
        loss_loader = torch.utils.data.DataLoader(data, batch_size, shuffle=True, generator=gen)
        
        for iteration, (f_inputs, f_labels) in enumerate(loss_loader):
            
            if max_iter is not None and iteration == max_iter:
                break
            current_time = timeit.default_timer()
            history['w'].append(deepcopy(net.state_dict()))
            history['time'].append(current_time - run_start)
            history['n_samples'].append(batch_size*3)
            if max_runtime > 0 and current_time - run_start >= max_runtime:
                break
            
            
            
            ########################
            ## UPDATE MULTIPLIERS ##
            ########################
            net.zero_grad()
            slack_vars.grad = None
            
            # sample for and calculate constraints (lines 2, 3)
            c_sample = [ci.sample_loader() for ci in constraints]
            c_1 = [ci.eval(net, c_sample[i]).reshape(1) + slack_vars[i] for i, ci in enumerate(c)]
            # update multipliers (line 3)
            with torch.no_grad():
                _lambda = _lambda + eta * torch.concat(c_1)
            # dual safeguard (lines 4,5)
            if torch.norm(_lambda) >= lambda_bound:
                _lambda = torch.zeros_like(_lambda, requires_grad=True)
                
                
                
            #######################
            ## UPDATE PARAMETERS ##
            #######################
            outputs = net(f_inputs)
            if f_labels.dim() < outputs.dim():
                f_labels = f_labels.unsqueeze(1)
            loss_eval = loss_fn(outputs, f_labels)
            loss_eval.backward() # loss grad
            f_grad = net_grads_to_tensor(net)
            f_grad = torch.concat([f_grad, torch.zeros(m)]) # add zeros for slack vars
            net.zero_grad()
            # constraint grad estimate
            c_grad = []
            for ci in c_1:
                ci.backward()
                ci_grad = net_grads_to_tensor(net)
                c_grad.append(torch.concat([ci_grad, slack_vars.grad]))
                net.zero_grad()
                slack_vars.grad = None
            c_grad = torch.stack(c_grad)
            # independent constraint estimate
            with torch.no_grad():
                c_sample = [ci.sample_loader() for ci in constraints]
                c_2 = torch.concat([ci.eval(net, c_sample[i]).reshape(1) + slack_vars[i] for i, ci in enumerate(c)])
            x_t = torch.concat([
                net_params_to_tensor(net, flatten=True, copy=True),
                slack_vars
            ])
            G = f_grad + c_grad.T @ _lambda + rho*(c_grad.T @ c_2) + mu*(x_t - z)
            x_t1 = project(x_t - tau*G, m)
            z += beta*(x_t-z)
            with torch.no_grad():
                _set_weights(net, x_t1)
                for i in range(len(slack_vars)):
                    slack_vars[i] = x_t1[i-len(slack_vars)]
            with np.printoptions(precision=6, suppress=True, floatmode='fixed'):
                print(f"""{iteration:5}|{loss_eval.detach().cpu().numpy()}|{_lambda.detach().cpu().numpy()}|{c_2.detach().cpu().numpy()}|{slack_vars.detach().cpu().numpy()}""", end='\r')
        
    ######################
    ### POSTPROCESSING ###    
    ######################
    
    G_hat = torch.zeros_like(G)
    
    f_inputs, f_labels = data[:][0], data[:][1]
    cgrad_sample = [ci.sample_dataset(np.inf) for ci in constraints]
    c_sample = [ci.sample_dataset(np.inf) for ci in constraints]

    net.zero_grad()
    slack_vars.grad = None
    # loss
    outputs = net(f_inputs)       
    if f_labels.dim() < outputs.dim():
        f_labels = f_labels.unsqueeze(1)
    loss_eval = loss_fn(outputs, f_labels)
    # loss grad
    loss_eval.backward()
    f_grad = net_grads_to_tensor(net)
    f_grad = torch.concat([f_grad, torch.zeros(m)]) # add zeros for slack vars
    net.zero_grad()
    # constraint grad estimate
    c_1 = [ci.eval(net, c_sample[i]).reshape(1) + slack_vars[i] for i, ci in enumerate(c)]
    c_grad = []
    for ci in c_1:
        ci.backward()
        ci_grad = net_grads_to_tensor(net)
        c_grad.append(torch.concat([ci_grad, slack_vars.grad]))
        net.zero_grad()
        slack_vars.grad = None
    c_grad = torch.stack(c_grad)
    
    # independent constraint estimate
    with torch.no_grad():
        c_2 = torch.concat([
            ci.eval(net, cgrad_sample[i]).reshape(1) + slack_vars[i] for i, ci in enumerate(c)
        ])
    x_t = torch.concat([
        net_params_to_tensor(net, flatten=True, copy=True),
        slack_vars
    ])
    G_hat += f_grad + c_grad.T @ _lambda + rho*(c_grad.T @ c_2) + mu*(x_t - z)
        
    x_t1 = project(x_t - tau*G_hat, m)
    with torch.no_grad():
        _set_weights(net, x_t1)
    
    current_time = timeit.default_timer()
    history['w'].append(deepcopy(net.state_dict()))
    history['time'].append(current_time - run_start)
    history['n_samples'].append(batch_size*3)
    
    return history


def _set_weights(net: torch.nn.Module, x):
    start = 0
    w = net_params_to_tensor(net, flatten=False, copy=False)
    for i in range(len(w)):
        end = start + w[i].numel()
        w[i].set_(x[start:end].reshape(w[i].shape))
        start = end