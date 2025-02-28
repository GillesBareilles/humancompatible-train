import pandas as pd
from IPython.display import Markdown, display
from sklearn.model_selection import train_test_split
import numpy as np
import os
import sys
import argparse
import importlib
import torch
from torch import tensor
from torch.utils.data import TensorDataset, DataLoader
# from ..optimize.stochastic_constraint import StochasticConstraint

def printmd(string):
    display(Markdown(string))

# from sklearn.preprocessing import StandardScaler
from utils.datasets.load_compas import load_compas

sys.path.append("..")  # Add parent directory to the sys.path
pd.options.mode.chained_assignment = None


RACE_IND = 4
SENSITIVE_CODE_0 = 0
SENSITIVE_CODE_1 = 1


def eq_loss_constr(loss, net, c_data, loss_bound):
    w_inputs, w_labels = c_data[0]
    b_inputs, b_labels = c_data[1]
    w_outs = net(w_inputs)
    w_loss = loss(w_outs, w_labels)
    b_outs = net(b_inputs)
    b_loss = loss(b_outs, b_labels)

    return torch.abs(w_loss - b_loss) - loss_bound


if __name__ == "__main__":
    
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

    from train.optimize.auglagr import ALOptimizer
    from train.optimize.stochastic_constraint import StochasticConstraint
    
    ## prepare data ##
    x_train, X_train, y_train, X_val, y_val = load_compas()
    print('Data loaded')

    ip_size = x_train.shape[1]

    x_black = X_train[(x_train[:, RACE_IND] == SENSITIVE_CODE_1), :]
    y_black = y_train[(x_train[:, RACE_IND] == SENSITIVE_CODE_1)]
    x_white = X_train[(x_train[:, RACE_IND] == SENSITIVE_CODE_0), :]
    y_white = y_train[(x_train[:, RACE_IND] == SENSITIVE_CODE_0)]
    
    X_train = np.concatenate([x_white, x_black[:160]])
    y_train = np.concatenate([y_white, y_black[:160]])

    train_ds = TensorDataset(tensor(X_train[:, :ip_size], dtype=torch.float),tensor(y_train, dtype=torch.float))
    train_loader = DataLoader(train_ds, batch_size=16)
    
    con_ds_b = TensorDataset(tensor(x_black[:160, :ip_size], dtype=torch.float),tensor(y_black[:160], dtype=torch.float))
    con_loader_b = DataLoader(con_ds_b, batch_size=16)
    con_ds_w = TensorDataset(tensor(x_white[:, :ip_size], dtype=torch.float),tensor(y_white, dtype=torch.float))
    con_loader_w = DataLoader(con_ds_w, batch_size=16)

    trials = 5
    acc_arr = []
    max_acc = 0
    ftrial, ctrial = [], []
    initsaved = []
    saved_model = []
    hid_size1 = 16
    hid_size2 = 16
    op_size = 1
    layer_sizes = [ip_size, hid_size1, hid_size2, op_size]
    model_specs = (layer_sizes,)
    
    sampling_int = 'exp'
    # sampling_int = 1
    loss_bound = 1e-2

    for trial in range(trials):
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>TRIAL", trial)
               
        from torch.nn import BCELoss
        loss = BCELoss()
        constr_loader = [con_loader_w, con_loader_b]
        net = CustomNetwork(model_specs)

        # TODO: change lambda to kwargs
        st_constraint_fn=lambda net, d: eq_loss_constr(loss, net, d, loss_bound)
        
        eq_c = StochasticConstraint(fun=st_constraint_fn, data_source=[con_loader_b, con_loader_w], batch_size=16)


        alo = ALOptimizer(net, loss, m_det=0, m_st=1, st_constraint_fn=eq_c, lr=1e-4, t=1.5)
        alo.optimize_class(train_loader, maxiter=50, epochs=10, constr_sampling_interval=sampling_int)

        # alo = ALOptimizer(net, loss, m_det = 0, m_st=1, st_constraint_fn=lambda net,
        #                   d: eq_loss_constr(loss, net, d, loss_bound), lr = 5e-4, t=2)
        # alo.optimize_(train_loader, constr_loader, maxiter=20, epochs=5, constr_sampling_interval=sampling_int)
        

        ## save history ##
        w = [param for param in net.parameters()]
        iterfs = alo.history['loss']
        itercs = alo.history['constr']
        if np.any([layer.isnan().any() for layer in w]):
            print("reached infeasibility not saving the model")
        else:
            ftrial.append(iterfs)
            ctrial.append(itercs)

            saved_models_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils', 'saved_models'))
            directory = os.path.join(saved_models_path, 'compas', model_name)

            if not os.path.exists(directory):
                os.makedirs(directory)

            # Save the model
            model_path = os.path.join(directory, f'net_al_compas_lb{loss_bound}_s{sampling_int}_tr{trial}')
            net.save_model(model_path)
        
    ftrial = np.array(ftrial).T
    ctrial = np.array(ctrial).T
    print(">>>>>>>>>>>>>>>>>>>Completed trials<<<<<<<<<<<<<<<<")
    print(f'Avg loss: {np.mean(ftrial)}')
    print(f'Avg с: {np.mean(ctrial)}')
    #print(acc_arr)
    df_ftrial = pd.DataFrame(ftrial, columns=range(1, ftrial.shape[1]+1), index=range(1, ftrial.shape[0]+1))
    df_ctrial = pd.DataFrame(ctrial, columns=range(1, ctrial.shape[1]+1), index=range(1, ctrial.shape[0]+1))

    # Save DataFrames to CSV files
    utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils', 'exp_results'))
    if not os.path.exists(utils_path):
        os.makedirs(utils_path)
    df_ftrial.to_csv(os.path.join(utils_path, 'l_compas_ftrial_'+str(loss_bound)+'.csv'))
    df_ctrial.to_csv(os.path.join(utils_path, 'l_compas_ctrial_'+str(loss_bound)+'.csv'))