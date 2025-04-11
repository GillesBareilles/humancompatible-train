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


def SSLPD(net: torch.nn.Module, data, w_ind, b_ind, loss_bound,
            batch_size=1,
            max_runtime=np.inf,
            lambda_bound = 100,
            rho = 5,
            mu = 2.,
            tau = 1e-4,
            beta = 0.1,
            eta = 1e-3,
            B = 1000,
            start_lambda=None,
            device='cpu',
            epochs=1,
            seed = 42,
            max_iter = None):
        
    history = {'loss': [],
               'constr': [],
               'w': [],
               'time': []}
    
    # slack variables
    slack_vars = torch.zeros(2, requires_grad=True)
    
    c1 = lambda net, d, s: one_sided_loss_constr(loss_fn, net, d) - loss_bound + s
    c2 = lambda net, d, s: -one_sided_loss_constr(loss_fn, net, d) - loss_bound + s
    
    c = [c1, c2]
    
    data_w = torch.utils.data.Subset(data, w_ind)
    data_b = torch.utils.data.Subset(data, b_ind)
    
    loss_fn = torch.nn.BCEWithLogitsLoss()
    n = sum(p.numel() for p in net.parameters())
    
    _lambda = torch.zeros(len(c),requires_grad=True) if start_lambda is None else start_lambda
    z = torch.concat([
            net_params_to_tensor(net, flatten=True, copy=True),
            slack_vars
        ])
    
    run_start = timeit.default_timer()
    for epoch in range(epochs):
        
        gen = torch.Generator(device=device)
        gen.manual_seed(seed+epoch)
        loader = torch.utils.data.DataLoader(data, batch_size, shuffle=True, generator=gen)
        loader_w = cycle(torch.utils.data.DataLoader(data_w, batch_size, shuffle=True, generator=gen))
        loader_b = cycle(torch.utils.data.DataLoader(data_b, batch_size, shuffle=True, generator=gen))
        
        for iteration, (f_inputs, f_labels) in enumerate(loader):
            
            if max_iter is not None and iteration == max_iter:
                return history
            current_time = timeit.default_timer()
            history['time'].append(current_time - run_start)
            if max_runtime > 0 and current_time - run_start >= max_runtime:
                return history
            
            ########################
            ## UPDATE MULTIPLIERS ##
            ########################
            
            net.zero_grad()
            slack_vars.grad = None
            
            # sample for constraints (line 2)
            cw_sample = next(loader_w)
            cb_sample = next(loader_b)
            c_sample = [cw_sample, cb_sample]
            # calc constraints and update multipliers (line 3)
            c_1 = torch.concat([
                ci(net, c_sample, slack_vars[i]).reshape(1) for i, ci in enumerate(c)
            ])
            _lambda += eta * c_1
            
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
            f_grad = torch.concat([f_grad, torch.zeros(len(c))]) # add zeros for slack vars
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
            cw_sample = next(loader_w)
            cb_sample = next(loader_b)
            c_sample = [cw_sample, cb_sample]
            with torch.no_grad():
                c_2 = torch.concat([ci(net, c_sample, slack_vars[i]).reshape(1) for i, ci in enumerate(c)])
            
            x_t = torch.concat([
                net_params_to_tensor(net, flatten=True, copy=True),
                slack_vars
            ])
            
            G = f_grad + c_grad.T @ _lambda + rho*(c_grad.T @ c_2) + mu*(x_t - z)

            x_t1 = project(x_t - tau*G, 2)
            z += beta*(x_t-z)
            
            with torch.no_grad():
                _set_weights(net, x_t1)
                for i in range(len(slack_vars)):
                    slack_vars[i] = x_t1[i-len(slack_vars)]
                    
            print(f"""{iteration}|{loss_eval.detach().cpu().numpy()}|{_lambda.detach().cpu().numpy()}|{c_1.detach().cpu().numpy()}|{slack_vars.detach().cpu().numpy()}""", end='\r')
            history['w'].append(deepcopy(net.state_dict()))
        
    ######################
    ### POSTPROCESSING ###    
    ######################
    
    G_hat = torch.zeros(1)
    
    for iteration, (f_inputs, f_labels) in enumerate(loader):
        cgrad_sample = [next(loader_w), next(loader_b)]
        c_sample = [next(loader_w), next(loader_b)]

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
        f_grad = torch.concat([f_grad, torch.zeros(2)]) # add zeros for slack vars
        net.zero_grad()
        # constraint grad estimate
        c_1 = torch.concat([
            ci(net, c_sample, slack_vars[i]).reshape(1) for i, ci in enumerate(c)
        ])
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
                ci(net, cgrad_sample, slack_vars[i]).reshape(1) for i, ci in enumerate(c)
            ])
        x_t = torch.concat([
            net_params_to_tensor(net, flatten=True, copy=True),
            slack_vars
        ])
        G_hat += f_grad + c_grad.T @ _lambda + rho*(c_grad.T @ c_2) + mu*(x_t - z)
        
    x_t1 = project(x_t - tau*G_hat, 2)
    
    _set_weights(net, x_t1)
    history['w'].append(deepcopy(net.state_dict()))
    
    return history


def _set_weights(net: torch.nn.Module, x):
    w = net_params_to_tensor(net, flatten=False, copy=False)
    for i in range(len(w)):
        end = start + w[i].numel()
        w[i].set_(x[start:end].reshape(w[i].shape))
        start = end