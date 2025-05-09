import argparse
from itertools import product
import os
import sys
import pandas as pd
from utils.load_folktables import load_folktables_torch
import numpy as np
import torch
from torch import tensor, nn
from torch.utils.data import TensorDataset, DataLoader

parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(parent_dir)))
        
from src.algos.constraints import one_sided_loss_constr
from src.algos.sslpd import SSLPD
from src.algos.sw_sub import SwitchingSubgradient_unbiased
from src.algos.auglag import *
from src.algos.ghost import StochasticGhost

class SimpleNet(nn.Module):
    def __init__(self, in_shape, out_shape, dtype):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_shape, 64, dtype=dtype),
            nn.ReLU(),
            nn.Linear(64, 32, dtype=dtype),
            nn.ReLU(),
            nn.Linear(32, out_shape, dtype=dtype),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
    
class SimpleDiffNet(nn.Module):
    def __init__(self, in_shape, out_shape, dtype):
        super().__init__()
        self.linear_gelu_stack = nn.Sequential(
            nn.Linear(in_shape, 64, dtype=dtype),
            nn.GELU(),
            nn.Linear(64, 32, dtype=dtype),
            nn.GELU(),
            nn.Linear(32, out_shape, dtype=dtype),
        )

    def forward(self, x):
        logits = self.linear_gelu_stack(x)
        return logits
    


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
                    prog='folktables exp')
    
    ### experiment parameters
    parser.add_argument('-ne', '--num_exp', type=int)
    parser.add_argument('-task', '--task', type=str)
    parser.add_argument('-state', '--state', type=str)
    parser.add_argument('-loss_bound', '--loss_bound', type=float)
    parser.add_argument('-constraint', '--constraint', type=str)
    parser.add_argument('-alpha', '--geomp', nargs='?', const=0.3, default=0.3, type=float)
    
    ### algorithm parameters
    # parser.add_argument('-maxiter', '--maxiter', nargs='?', const=None, type=int)
    # parser.add_argument('-stepsize', '-sr', nargs='?', const='inv_iter', default='inv_iter', type=str)
    # parse args
    args = parser.parse_args()
    ALG_TYPE = 'sg'
    EXP_NUM = args.num_exp
    FT_STATE = args.state
    LOSS_BOUND = args.loss_bound
    CONSTRAINT = args.constraint
    TASK = args.task
    
    
    rhos = [0.2, 0.8, 1, 10]
    betas = [5, 10, 20]
    lambdas = [0.1, 0.5, 0.9]
    gamma0s = [0.005, 0.05, 0.1, 0.5]
    zetas = [0.001, 0.005, 0.01]
    taus = [0.5, 1, 5]

    param_grid = product(rhos, betas, lambdas, gamma0s, zetas, taus)

    for rho, beta, ghost_lambda, gamma0, zeta, tau in param_grid:
        G_ALPHA = args.geomp
        MAXITER_GHOST = 1000
        ghost_rho = rho
        ghost_beta = beta
        ghost_lambda = ghost_lambda
        ghost_gamma0 = gamma0
        ghost_zeta = zeta
        ghost_tau = tau
        ghost_stepsize_rule = 'dimin'
        params_str = f'a{G_ALPHA}rho{ghost_rho}beta{ghost_beta}lambda{ghost_lambda}gamma{ghost_gamma0}zeta{ghost_zeta}tau{ghost_tau}ss{ghost_stepsize_rule}'
        
        ALG_CUSTOM_NAME = params_str
        ALG_TYPE = 'sg_' + ALG_CUSTOM_NAME

        device = 'cpu'
        torch.set_default_device(device)
        
        DTYPE = torch.float32

        FT_DATASET = TASK
        torch.set_default_dtype(DTYPE)
        DATASET_NAME = FT_DATASET + '_' + FT_STATE
        
        X_train, y_train, [w_idx_train, nw_idx_train], X_test, y_test, [w_idx_test, nw_idx_test] = load_folktables_torch(
            FT_DATASET, state=FT_STATE.upper(), random_state=42, make_unbalanced = False, onehot=False
        )
            
        X_train_tensor = tensor(X_train, dtype=DTYPE)
        y_train_tensor = tensor(y_train, dtype=DTYPE)
        train_ds = TensorDataset(X_train_tensor,y_train_tensor)
        print(f'Train data loaded: {(FT_DATASET, FT_STATE)}')
        print(f'Data shape: {X_train_tensor.shape}')
        
        read_model = False
        
        if CONSTRAINT == 'fpr':
            statistic = FalsePositiveRate()
        elif CONSTRAINT == 'pr':
            statistic = PositiveRate()
        else:
            statistic = None
        
        saved_models_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils', 'saved_models'))
        directory = os.path.join(saved_models_path, DATASET_NAME,CONSTRAINT,f'{LOSS_BOUND:.0E}')
        if ALG_TYPE.startswith('sg'):
            model_name = os.path.join(directory, f'{ALG_TYPE}_{LOSS_BOUND}_p{G_ALPHA}')
        else:
            model_name = os.path.join(directory, f'{ALG_TYPE}_{LOSS_BOUND}')
            
        
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        ftrial, ctrial, wtrial, ttrial, samples_trial = [], [], [], [], []
        
        # experiment loop
        for EXP_IDX in range(EXP_NUM):
            
            # torch.manual_seed(EXP_IDX)
            model_path = model_name + f'_trial{EXP_IDX}.pt'
            
            net = SimpleNet(in_shape=X_test.shape[1], out_shape=1, dtype=DTYPE).to(device)
            if read_model:
                net.load_state_dict(torch.load(model_path, weights_only=False, map_location=torch.device('cpu')))
            
            N = min(len(w_idx_train), len(nw_idx_train))
            
            history = StochasticGhost(net, train_ds, w_idx_train, nw_idx_train,
                                geomp=G_ALPHA,
                                stepsize_rule=ghost_stepsize_rule,
                                zeta = ghost_zeta,
                                gamma0 = ghost_gamma0,
                                beta=ghost_beta,
                                rho=ghost_rho,
                                lamb = ghost_lambda,
                                tau = ghost_tau,
                                loss_bound=LOSS_BOUND,
                                maxiter=MAXITER_GHOST,
                                seed=EXP_IDX)
            ## SAVE RESULTS ##
            ftrial.append(pd.Series(history['loss']))
            ctrial.append(pd.DataFrame(history['constr']))
            wtrial.append(history['w'])
            ttrial.append(history['time'])
            samples_trial.append(pd.Series(history['n_samples']))
            
            # Save the model
            
            torch.save(net.state_dict(), model_path)
            print('')
        
        # Save DataFrames to CSV files
        utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils', 'exp_results'))
        if not os.path.exists(utils_path):
            os.makedirs(utils_path)
        
        ftrial = pd.concat(ftrial, keys=range(len(ftrial)))
        ctrial = pd.concat(ctrial, keys=range(len(ctrial)))
        samples_trial = pd.concat(samples_trial, keys=range(len(samples_trial)))
        
        if ALG_TYPE.startswith('sg'):
            fname = f'{ALG_TYPE}_{DATASET_NAME}_{LOSS_BOUND}_{G_ALPHA}'
        else:
            fname = f'{ALG_TYPE}_{DATASET_NAME}_{LOSS_BOUND}'
        print(f'Saving to: {fname}')
        ftrial.to_csv(os.path.join(utils_path, fname + '_ftrial.csv'))
        ctrial.to_csv(os.path.join(utils_path, fname + '_ctrial.csv'))
        samples_trial.to_csv(os.path.join(utils_path, fname + '_samples.csv'))
        
        print('----')
        # df(n_iter, n_trials)
        wlen = max([len(tr) for tr in wtrial])
        index = pd.MultiIndex.from_product([['train', 'test'], np.arange(wlen), np.arange(EXP_NUM)], names=('is_train', 'iteration', 'trial'))
        full_stats = pd.DataFrame(index=index, columns=['Loss', 'C1', 'C2', 'SampleSize', 'time'])
        full_stats.sort_index(inplace=True)
        
        net = SimpleNet(in_shape=X_test.shape[1], out_shape=1, dtype=DTYPE).to(device)
        loss_fn = nn.BCEWithLogitsLoss()
        
        X_test_tensor = tensor(X_test, dtype=DTYPE).to(device)
        y_test_tensor = tensor(y_test, dtype=DTYPE).to(device)
        
        X_test_w = X_test_tensor[w_idx_test]
        y_test_w = y_test_tensor[w_idx_test]
        X_test_nw = X_test_tensor[nw_idx_test]
        y_test_nw = y_test_tensor[nw_idx_test]
        
        X_train_w = X_train_tensor[w_idx_train]
        y_train_w = y_train_tensor[w_idx_train]
        X_train_nw = X_train_tensor[nw_idx_train]
        y_train_nw = y_train_tensor[nw_idx_train]
        
        save_train = True
        
        TEST_SKIP_ITERS = 1
        with torch.inference_mode():
            for exp_idx in range(EXP_NUM):
                weights_to_eval = wtrial[exp_idx][::TEST_SKIP_ITERS]
                for alg_iteration, w in enumerate(weights_to_eval):
                    
                    if CONSTRAINT == 'loss':
                        c_f = one_sided_loss_constr
                        c_loss_fn = nn.BCEWithLogitsLoss()
                    elif CONSTRAINT == 'fpr':
                        c_f = fairret_constr
                        statistic = FalsePositiveRate()
                        c_loss_fn = NormLoss(statistic)
                    elif CONSTRAINT == 'pr':
                        c_f = fairret_pr_constr
                        statistic = PositiveRate()
                        c_loss_fn = NormLoss(statistic)
                    print(f'{exp_idx} | {alg_iteration}', end='\r')
                    net.load_state_dict(w)
                    
                    if save_train:
                        outs = net(X_train_tensor)
                        if y_train_tensor.ndim < outs.ndim:
                            y_train_tensor = y_train_tensor.unsqueeze(1)
                        loss = loss_fn(outs, y_train_tensor).detach().cpu().numpy()
                        
                        c1 = c_f(c_loss_fn, net, [(X_train_w, y_train_w), (X_train_nw, y_train_nw)]).detach().cpu().numpy()
                        c2 = -c1
                        # pandas multiindex bug(?) workaround
                        full_stats.loc['train'].at[alg_iteration, exp_idx] = {
                        'Loss': loss,
                        'C1': c1,
                        'C2': c2,
                        'SampleSize': samples_trial[exp_idx][alg_iteration],
                        'time': ttrial[exp_idx][alg_iteration]}
                        
                    outs = net(X_test_tensor)
                    if y_test_tensor.ndim < outs.ndim:
                        y_test_tensor = y_test_tensor.unsqueeze(1)
                    loss = loss_fn(outs, y_test_tensor).detach().cpu().numpy()
                    
                    c1 = c_f(c_loss_fn, net, [(X_test_w, y_test_w), (X_test_nw, y_test_nw)]).detach().cpu().numpy()
                    c2 = -c1
                    
                    full_stats.loc['test'].at[alg_iteration, exp_idx] = {
                        'Loss': loss,
                        'C1': c1,
                        'C2': c2,
                        'SampleSize': samples_trial[exp_idx][alg_iteration],
                        'time': ttrial[exp_idx][alg_iteration]}
                
        if ALG_TYPE.startswith('sg'):
            fname = f'{ALG_TYPE}_{DATASET_NAME}_{LOSS_BOUND}_{G_ALPHA}.csv'
        else:
            fname = f'{ALG_TYPE}_{DATASET_NAME}_{LOSS_BOUND}.csv'
        print(f'Saving to: {fname}')
        full_stats.to_csv(os.path.join(utils_path, fname))