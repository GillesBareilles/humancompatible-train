import pandas as pd
from IPython.display import Markdown, display
from sklearn.model_selection import train_test_split
import numpy as np
import os
import sys
import argparse
import importlib
import torch

from utils.datasets.load_compas import load_compas

def printmd(string):
    display(Markdown(string))

sys.path.append("..")  # Add parent directory to the sys.path
pd.options.mode.chained_assignment = None

RACE_IND = 4
SENSITIVE_CODE_0 = 0
SENSITIVE_CODE_1 = 1

class Operations:

    # For now the input data is passed as init parameters
    def __init__(self, data, net):

        self.x_train = data[0]
        self.y_train = data[1]
        self.x_val = data[2]
        self.y_val = data[3]
        self.x_train_raw = data[4]
        self.model = net

        black_idx = (self.x_train_raw[:, RACE_IND] == SENSITIVE_CODE_1)
        self.x_black = self.x_train[black_idx, :]
        self.y_black = self.x_train[black_idx]

        white_idx = (self.x_train_raw[:, RACE_IND] == SENSITIVE_CODE_0)
        self.x_white = self.x_train[white_idx, :]
        self.y_white = self.x_train[white_idx]

    def obj_fun(self, params, minibatch):
        x = self.x_train
        y = self.y_train
        model = self.model
        samples = np.random.choice(len(y), minibatch, replace=False)
        fval = model.get_obj(x[samples, :], y[samples], params)
        return fval

    def obj_grad(self, params, minibatch):
        fgrad = []
        x = self.x_train
        y = self.y_train
        model = self.model
        samples = np.random.choice(len(y), minibatch, replace=False)
        fgrad = model.get_obj_grad(x[samples, :], y[samples], params)

        return fgrad

    def conf1(self, params, minibatch):
        model = self.model
        
        black_batch_idxs = np.random.choice(len(self.y_black), minibatch, replace=False)
        white_batch_idxs = np.random.choice(len(self.y_white), minibatch, replace=False)
   
        conf1 = model.get_constraint(
            self.x_black[black_batch_idxs, :], self.y_train[black_batch_idxs],
            self.x_white[white_batch_idxs, :], self.y_train[white_batch_idxs],
            params)
        return conf1
    
    def conJ1(self, params, minibatch):
        model = self.model

        black_batch_idxs = np.random.choice(len(self.y_black), minibatch, replace=False)
        white_batch_idxs = np.random.choice(len(self.y_white), minibatch, replace=False)

        conj1 = model.get_constraint_grad(
            self.x_black[black_batch_idxs, :], self.y_train[black_batch_idxs],
            self.x_white[white_batch_idxs, :], self.y_train[white_batch_idxs],
            params)
        
        return conj1

    def conf2(self, params, minibatch):
        model = self.model

        black_batch_idxs = np.random.choice(len(self.y_black), minibatch, replace=False)
        white_batch_idxs = np.random.choice(len(self.y_white), minibatch, replace=False)

        conf2 = model.get_constraint(
            self.x_black[black_batch_idxs, :], self.y_train[black_batch_idxs],
            self.x_white[white_batch_idxs, :], self.y_train[white_batch_idxs],
            params)
        
        return conf2
    
    def conJ2(self, params, minibatch):
        model = self.model

        black_batch_idxs = np.random.choice(len(self.y_black), minibatch, replace=False)
        white_batch_idxs = np.random.choice(len(self.y_white), minibatch, replace=False)

        conj2 = model.get_constraint_grad(
            self.x_black[black_batch_idxs, :], self.y_train[black_batch_idxs],
            self.x_white[white_batch_idxs, :], self.y_train[white_batch_idxs],
            params)
        
        return conj2
    
def paramvals(maxiter, beta, rho, lamb, hess, tau, mbsz, numcon, geomp, stepdecay, gammazero, zeta, N, n, lossbound, scalef):
    params = {
        'maxiter': maxiter,  # number of iterations performed
        'beta': beta,  # trust region size
        'rho': rho,  # trust region for feasibility subproblem
        'lamb': lamb,  # weight on the subfeasibility relaxation
        'hess': hess,  # method of computing the Hessian of the QP, options include 'diag' 'lbfgs' 'fisher' 'adamdiag' 'adagraddiag'
        'tau': tau,  # parameter for the hessian
        'mbsz': mbsz,  # the standard minibatch size, used for evaluating the progress of the objective and constraint
        'numcon': numcon,  # number of constraint functions
        'geomp': geomp,  # parameter for the geometric random variable defining the number of subproblem samples
        # strategy for step decrease, options include 'dimin' 'stepwise' 'slowdimin' 'constant'
        'stepdecay': stepdecay,
        'gammazero': gammazero,  # initial stepsize
        'zeta': zeta,  # parameter associated with the stepsize iteration
        'N': N,  # Train/val sample size
        'n': n,  # Total number of parameters
        'lossbound': lossbound, #Bound on constraint loss
        'scalef': scalef #Scaling factor for constraints
    }
    return params



class Net(torch.nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.layer_norms = torch.nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i+1]))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu((layer(x)))
        x = torch.sigmoid(self.layers[-1](x))
        return x

if __name__ == "__main__":

    save_data = True

    x_train, X_train, y_train, X_val, y_val = load_compas()

    ## add bias to data
    white_idx = (x_train[:, RACE_IND] == SENSITIVE_CODE_1)
    black_idx = (x_train[:, RACE_IND] == SENSITIVE_CODE_0)
    x_black = X_train[black_idx, :]
    y_black = y_train[black_idx]
    x_white = X_train[white_idx, :]
    y_white = y_train[white_idx]
    
    X_train = np.concatenate([x_white, x_black[:160]])
    y_train = np.concatenate([y_white, y_black[:160]])

    x_train = np.concatenate((x_train[white_idx], x_train[black_idx][:160]))

    parser = argparse.ArgumentParser(description="Dynamically import the model class")

    # Add argument for module name
    parser.add_argument("--model", type=str, help="Name of the model to import (backend_connect)")

    parser.add_argument("--optimizer", type=str, help="Optimizer Name (Default StochasticGhost)")

    # Parse command-line arguments
    args = parser.parse_args()
    model_name = args.model
    optimizer_name = args.optimizer

    parent_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(os.path.dirname(parent_dir)))

    # Dynamically import the specified module
    if not model_name:
        # Import the default module if no module name is provided
        print("Architecture not specified, defaulting to PyTorch ('pytorch_connect')")
        model_name = 'pytorch_connect'

    if model_name == 'pytorch_connect':
        from train.connect.pytorch_connect import CustomNetwork

    from train.optimize.stochastic_ghost import StochasticGhost
    
    loss_bound=1e-2
    trials = 5
    maxiter = 500
    acc_arr = []
    max_acc = 0
    ftrial, ctrial1, ctrial2 = [], [], []
    initsaved = []
    #x_train, x_val, y_train, y_val = train_test_split(in_df.values, out_df.values, test_size=0.3, random_state=42)
    ip_size = x_train.shape[1]
    saved_model = []
    for trial in range(trials):
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>TRIAL", trial)
        
        #print(type(X_train))
        hid_size1 = 32
        hid_size2 = 32
        op_size = 1
        layer_sizes = [ip_size, hid_size1, hid_size2, op_size]
        
        data = (X_train[:, :ip_size], y_train, X_val[:, :ip_size], y_val, x_train)
        model_specs = (layer_sizes,)
        #x_len = x_train[:, 4]
        num_trials = min(len(y_train[((x_train[:, RACE_IND]) == SENSITIVE_CODE_1)]), len(y_train[(x_train[:, RACE_IND] == SENSITIVE_CODE_0)]))

        net = CustomNetwork(Net(layer_sizes), loss=torch.nn.BCELoss())
        operations = Operations(data, net)
        
        initw, num_param = net.get_trainable_params()
        params = paramvals(maxiter=maxiter, beta=10., rho=1e-3, lamb=0.5, hess='diag', tau=2., mbsz=100,
                        numcon=2, geomp=0.2, stepdecay='dimin', gammazero=0.1, zeta=0.7, N=num_trials, n=num_param, lossbound=[loss_bound, loss_bound], scalef=[1., 1.])
        solver_params = {'max_iter': 400, 'eps_abs': 1e-9, 'polish': True, 'eps_prim_inf': 1e-6, 'eps_dual_inf': 1e-6}
        w, iterfs, itercs = StochasticGhost(operations.obj_fun, operations.obj_grad, [operations.conf1, operations.conf2], [operations.conJ1, operations.conJ2],
                                            initw, params, solver_params=solver_params)
        
        if np.isnan(w[0]).any():
            print("reached infeasibility not saving the model")
        else:
            ftrial.append(iterfs)
            ctrial1.append(itercs[:,0])
            ctrial2.append(itercs[:,1])

            state_dict = net.state_dict()
            for k, val in zip(state_dict.keys(), w):
                state_dict[k] = torch.tensor(val)
            net.load_state_dict(state_dict)

            if save_data:
                saved_model.append(net)
                saved_models_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils', 'saved_models'))
                directory = os.path.join(saved_models_path, 'compas', model_name)

                # Check if the directory exists
                if not os.path.exists(directory):
                    # If the directory doesn't exist, create it
                    os.makedirs(directory)

                # Save the model
                model_path = os.path.join(directory, f'net_ghost_compas_lb{loss_bound}_tr{trial}')
                net.save_model(model_path)
    
    ftrial = np.array(ftrial).T
    ctrial1 = np.array(ctrial1).T
    ctrial2 = np.array(ctrial2).T
    print(">>>>>>>>>>>>>>>>>>>Completed trials<<<<<<<<<<<<<<<<")
    print(f'Avg loss: {np.mean(ftrial)}')
    print(f'Avg с1: {np.mean(ctrial1)}')
    print(f'Avg с2: {np.mean(ctrial2)}')
    #print(acc_arr)
    if save_data:
        df_ftrial = pd.DataFrame(ftrial, columns=range(1, ftrial.shape[1]+1), index=range(1, ftrial.shape[0]+1))
        df_ctrial1 = pd.DataFrame(ctrial1, columns=range(1, ctrial1.shape[1]+1), index=range(1, ctrial1.shape[0]+1))
        df_ctrial2 = pd.DataFrame(ctrial2, columns=range(1, ctrial2.shape[1]+1), index=range(1, ctrial2.shape[0]+1))

        # Save DataFrames to CSV files
        utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils', 'exp_results'))
        if not os.path.exists(utils_path):
            os.makedirs(utils_path)
        df_ftrial.to_csv(os.path.join(utils_path, 'compas_ftrial_'+str(loss_bound)+'.csv'))
        df_ctrial1.to_csv(os.path.join(utils_path, 'compas_ctrial1_'+str(loss_bound)+'.csv'))
        df_ctrial2.to_csv(os.path.join(utils_path, 'compas_ctrial2_'+str(loss_bound)+'.csv'))