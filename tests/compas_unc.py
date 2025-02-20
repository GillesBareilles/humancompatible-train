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
from torch.nn import BCELoss

def printmd(string):
    display(Markdown(string))

from sklearn.preprocessing import StandardScaler 

sys.path.append("..")  # Add parent directory to the sys.path
pd.options.mode.chained_assignment = None


RACE_IND = 4
SENSITIVE_CODE_0 = 0
SENSITIVE_CODE_1 = 1


def preprocess_data():

    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    file_path = os.path.join(data_dir, "compas-scores-two-years.csv")
    raw_data = pd.read_csv(file_path)

    df = raw_data[['age', 'c_charge_degree', 'race', 'age_cat', 'score_text', 'sex', 'priors_count',
                'days_b_screening_arrest', 'decile_score', 'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out']]
    df = df[(df['days_b_screening_arrest'] <= 30) & (df['days_b_screening_arrest'] >= -30) &
            (df['is_recid'] != -1) & (df['c_charge_degree'] != "O") & (df['score_text'] != 'N/A')]

    df['length_of_stay'] = pd.to_datetime(
        df['c_jail_out']) - pd.to_datetime(df['c_jail_in'])
    df['length_of_stay'] = df['length_of_stay'].dt.total_seconds() / 3600

    df_needed = df[(df['race'] == 'Caucasian') | (df['race'] == 'African-American')]
    race_mapping = {'African-American': SENSITIVE_CODE_1, 'Caucasian': SENSITIVE_CODE_0}

    # Create a new column 'race_code' based on the mapping
    df_needed['race_code'] = df_needed['race'].map(race_mapping)

    # Categorizing
    df_needed['crime_code'] = pd.Categorical(df_needed['c_charge_degree']).codes
    df_needed['age_code'] = pd.Categorical(df_needed['age_cat']).codes
    df_needed['race_code'] = df_needed['race'].map(race_mapping)
    df_needed['gender_code'] = pd.Categorical(df_needed['sex']).codes
    df_needed['score_code'] = pd.Categorical(df_needed['score_text']).codes
    df_needed['charge_degree_code'] = pd.Categorical(
        df_needed['c_charge_degree']).codes


    in_cols = ['priors_count', 'score_code', 'age_code', 'gender_code', 'race_code', 'crime_code', 'charge_degree_code']
    out_cols = ['two_year_recid']

    in_df = df_needed[in_cols]
    out_df = df_needed[out_cols]


    blacks_in = len(df_needed[(df_needed['race_code'] == SENSITIVE_CODE_1) & (df_needed['two_year_recid']== 0)])
    whites_in = len(df_needed[(df_needed['race_code'] == SENSITIVE_CODE_0) & (df_needed['two_year_recid'] == 0)])
    print(blacks_in, whites_in)


    x_train, x_val, y_train, y_val = train_test_split(in_df.values, out_df.values, test_size  = 0.30)

    # Normalization

    scaler = StandardScaler()  

    # Fitting only on training data
    scaler.fit(x_train)  
    X_train = scaler.transform(x_train)  

    # Applying same transformation to test data
    X_val = scaler.transform(x_val)

    # Assuming x_val and y_val are numpy arrays
    # Convert y_val to a column vector to match the shape of x_val
    #y_val = np.expand_dims(y_val, axis=1)

    # Concatenate x_val and y_val along the columns

    file_path_raw = '../data/val_data/val_data_raw_compas.csv'
    file_path_scaled = '../data/val_data/val_data_scaled_compas.csv'

    data_combined_raw = np.concatenate((x_val, y_val), axis=1)

    # Convert the combined data to a DataFrame
    df_combined = pd.DataFrame(data_combined_raw)

    #x_val_columns = ['priors_count', 'score_code', 'age_code', 'gender_code', 'race_code', 'crime_code', 'charge_degree_code']
    #y_val_columns = ['two_year_recid']

    if not os.path.exists(file_path_raw):
        df_combined.to_csv(file_path_raw, index=False)
        print("File saved successfully.")
    else:
        print("File already exists. Not saving again.")

    data_combined_scaled = np.concatenate((X_val, y_val), axis=1)
    df_combined = pd.DataFrame(data_combined_scaled)
    if not os.path.exists(file_path_scaled):
        df_combined.to_csv(file_path_scaled, index=False)
        print("File saved successfully.")
    else:
        print("File already exists. Not saving again.")    
    return  x_train, X_train, y_train, X_val, y_val


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
    sys.path.append(os.path.abspath(os.path.join(parent_dir, "../humancompatible/train")))

    # Dynamically import the specified module
    if not model_name:
        # Import the default module if no module name is provided
        print("Architecture not specified, defaulting to PyTorch ('pytorch_connect')")
        model_name = 'pytorch_connect'
    
    os.chdir(os.path.join(os.getcwd(), 'tests'))
    print(f'working in: {os.getcwd()}')

    model = importlib.import_module(model_name)
    CustomNetwork = getattr(model, "CustomNetwork")

    optimizer = importlib.import_module('auglagr')
    ALOptimizer = getattr(optimizer, 'ALOptimizer')
    
    ## prepare data ##
    x_train, X_train, y_train, X_val, y_val = preprocess_data()
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
    loss_bound = 1e-2
    epochs = 5*15
    loss = BCELoss()

    from torch.optim import Adam
    net = CustomNetwork(model_specs)
    opt = Adam(net.parameters())

    for epoch in range(epochs):
        for inputs, labels in train_loader:
            outs = net(inputs)
            l = loss(outs, labels)
            l.backward()
            opt.step()
            
    directory = "../saved_models/compas/" + model_name

    if not os.path.exists(directory):
        os.makedirs(directory)
    # Save the model
    model_path = os.path.join(directory, f'net_al_compas_uncon')
    net.save_model(model_path)
