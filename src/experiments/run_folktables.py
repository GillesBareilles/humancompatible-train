import argparse
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
    parser.add_argument('-alg', '--algorithm')
    parser.add_argument('-ne', '--num_exp', type=int)
    parser.add_argument('-task', '--task', type=str)
    parser.add_argument('-state', '--state', type=str)
    parser.add_argument('-loss_bound', '--loss_bound', type=float)
    parser.add_argument('-constraint', '--constraint', type=str)
    
    ### algorithm parameters
    parser.add_argument('-maxiter', '--maxiter', nargs='?', const=None, type=int)
    # ghost
    parser.add_argument('-alpha', '--geomp', nargs='?', const=0.2, default=0.2, type=float)
    parser.add_argument('-beta', '--beta', nargs='?', const=10, default=10, type=float)
    parser.add_argument('-rho', '--rho', nargs='?', const=0.8, default=0.8, type=float)
    parser.add_argument('-lambda', '--lam', nargs='?', const=0.5, default=0.5, type=float)
    parser.add_argument('-gamma0', '--gamma0', nargs='?', const=0.05, default=0.05, type=float)
    parser.add_argument('-zeta', '--zeta', nargs='?', const=0.3, default=0.3, type=float)
    parser.add_argument('-tau', '--tau', nargs='?', const=1, default=1, type=float)
    parser.add_argument('-stepsize', '-sr', nargs='?', const='inv_iter', default='inv_iter', type=str)
    
    
    # alm
    parser.add_argument('-bs', '--batch_size', nargs='?', const=16, default=16, type=int)
    
    # ssg
    parser.add_argument('-frule', '--frule', nargs='?', const='dimin', default='dimin', type=str)
    parser.add_argument('-fs', '--f_stepsize', nargs='?', const=7e-1, default=7e-1, type=float)
    parser.add_argument('-crule', '--crule', nargs='?', const='dimin', default='dimin', type=str)
    parser.add_argument('-cs', '--c_stepsize', nargs='?', const=7e-1, default=7e-1, type=float)
    parser.add_argument('-epochs', '--epochs', nargs='?', const=1, default=1, type=int)
    parser.add_argument('-ctol', '--ctol', nargs=1, type=float)
    
    # sslalm
    parser.add_argument('-mu', '--mu', nargs='?', const=2, default=2, type=float)
    
    # parse args
    args = parser.parse_args()
    ALG_TYPE = args.algorithm
    EXP_NUM = args.num_exp
    FT_STATE = args.state
    LOSS_BOUND = args.loss_bound
    CONSTRAINT = args.constraint
    TASK = args.task
    
    if ALG_TYPE == 'sg':
        G_ALPHA = args.geomp
        MAXITER_GHOST = 1000 if args.maxiter is None else args.maxiter
        ghost_rho = args.rho
        ghost_beta = args.beta
        ghost_lambda = args.lam
        ghost_gamma0 = args.gamma0
        ghost_zeta = args.zeta
        ghost_tau = args.tau
        ghost_stepsize_rule = args.stepsize
    elif ALG_TYPE == 'swsg':
        epochs=args.epochs
        ctol = args.ctol
        BATCH_SIZE = args.batch_size
        f_stepsize_rule=args.frule
        f_stepsize=args.f_stepsize
        c_stepsize_rule=args.crule
        c_stepsize=args.c_stepsize
    elif ALG_TYPE == 'aug':
        epochs=args.epochs
        BATCH_SIZE = args.batch_size
        MAXITER_ALM = 1000 if args.maxiter is None else args.maxiter
    elif ALG_TYPE == 'sslalm':
        epochs = args.epochs
        BATCH_SIZE = args.batch_size
        lambda_bound = 100
        rho = args.rho
        mu = args.mu
        tau = args.tau
        beta = args.beta
        eta = 5e-3
        
        
    if ALG_TYPE == 'sg':
        device = 'cpu'
        print('CUDA not supported for Stochastic Ghost')
    elif torch.cuda.is_available():
        device = 'cuda'
        print('CUDA found')
    else:
        device = 'cpu'
        print('CUDA not found')
    
    print(f'{device = }')    
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
    
    # TODO: move to command line args
    # EXP_NUM = 7
    RUNTIME_LIMIT = 15
    UPDATE_LAMBDA = True
    # G_ALPHA = 0.3
    # ALG_TYPE = 'sg'
    # BATCH_SIZE = 16
    # MAXITER_GHOST = 1500
    MAXITER_ALM = np.inf
    MAXITER_SSG = np.inf
    MAXITER_SSLALM = np.inf
    TEST_SKIP_ITERS = 1
    
    read_model = False
    
    saved_models_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils', 'saved_models'))
    directory = os.path.join(saved_models_path, DATASET_NAME,f'{LOSS_BOUND:.0E}')
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
        
        
        if ALG_TYPE.startswith('swsg'):
            history = SwitchingSubgradient_unbiased(net, train_ds, w_idx_train, nw_idx_train,
                                                   loss_bound = LOSS_BOUND,
                                                   batch_size = BATCH_SIZE,
                                                   epochs = epochs,
                                                   ctol = ctol,
                                                   f_stepsize_rule = f_stepsize_rule,
                                                   f_stepsize = f_stepsize,
                                                   c_stepsize_rule = c_stepsize_rule,
                                                   c_stepsize = c_stepsize,
                                                   device=device,
                                                   seed=EXP_IDX)
        elif ALG_TYPE.startswith('sg'):
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
        elif ALG_TYPE.startswith('aug') or ALG_TYPE == 'aug':
            history = AugLagr(net, train_ds,
                              w_idx_train,
                              nw_idx_train,
                              batch_size=BATCH_SIZE,
                              loss_bound=LOSS_BOUND,
                              update_lambda=UPDATE_LAMBDA,
                              maxiter=MAXITER_ALM,
                              device=device,
                              epochs=epochs,
                              seed=EXP_IDX)
        elif ALG_TYPE.startswith('sslalm'):
            history = SSLPD(net, train_ds, w_idx_train, nw_idx_train,
                            loss_bound=LOSS_BOUND,
                            epochs=epochs,
                            batch_size=BATCH_SIZE,
                            lambda_bound = 10.,
                            rho = rho,
                            mu = mu,
                            tau = tau,
                            beta = beta,
                            eta = 5e-2,
                            max_iter=MAXITER_SSLALM,
                            device=device,
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