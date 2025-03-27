import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import os
import sys
from sklearn.preprocessing import StandardScaler

sys.path.append("..")

from folktables import ACSDataSource, ACSPublicCoverage, ACSEmployment

RAC1P_WHITE = 1

def load_coverage_torch(dataset: str = 'employment', state='AL', random_state=None, make_unbalanced=False):
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'raw_data', dataset))
    data_source = ACSDataSource(survey_year=2018, horizon='1-Year', survey='person', root_dir=data_dir)
    acs_data = data_source.get_data(states=[state], download=True)
    if dataset == 'employment':
        features, label, group = ACSEmployment.df_to_numpy(acs_data)
    elif dataset == 'coverage':
        features, label, group = ACSPublicCoverage.df_to_numpy(acs_data)
    
    group_binary = (group == RAC1P_WHITE).astype(float)
        
    # stratify by binary race (white vs rest)
    X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
        features, label, group_binary, test_size=0.2, stratify = group_binary, random_state=random_state)
    
    if make_unbalanced:
        # g_train_new = g_train[:len(g_train)/2]
        train_w_idx = np.argwhere(g_train == 1).flatten()
        train_nw_idx = np.argwhere(g_train != 1).flatten()
        train_nw_idx = train_nw_idx[:len(train_nw_idx)//10]
        idx = np.concatenate([train_w_idx, train_nw_idx])
        X_train = X_train[idx]
        y_train = y_train[idx]
        g_train = g_train[idx]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    train_w_idx = np.argwhere(g_train == 1).flatten()
    train_nw_idx = np.argwhere(g_train != 1).flatten()
    
    test_w_idx = np.argwhere(g_test == 1).flatten()
    test_nw_idx = np.argwhere(g_test != 1).flatten()
    
    return X_train_scaled, y_train, [train_w_idx, train_nw_idx], X_test_scaled, y_test, [test_w_idx, test_nw_idx]
    
    

def load_coverage():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'raw_data', 'coverage'))
    data_source = ACSDataSource(survey_year=2018, horizon='1-Year', survey='person', root_dir=data_dir)
    acs_data = data_source.get_data(states=["CA"], download=True)

    ca_features, ca_labels, _ = ACSPublicCoverage.df_to_pandas(acs_data)

    ca_features_filt = ca_features[(ca_features['RAC1P'] == 1) | (ca_features['RAC1P'] == 2) | (ca_features['RAC1P'] == 6)]
    ca_labels_filt = ca_labels[(ca_features['RAC1P'] == 1) | (ca_features['RAC1P'] == 2) | (ca_features['RAC1P'] == 6)]
    ca_features_filt['RAC1P'] = ca_features_filt['RAC1P'].replace({1: 0, 2: 2, 6: 1})
    ca_features_filt['SEX'] = ca_features_filt['SEX'].replace({1: 0, 2: 1})

    # filterning based on non-numeric values
    ca_features_filt[ca_features_filt.select_dtypes(include='object').any(axis=1)]
    ca_labels_filt[ca_features_filt.select_dtypes(include='object').any(axis=1)]

    ###### Create coverage bins
    pincp_column = ca_features_filt["PINCP"]
    # Calculate bin size
    bin_size = (pincp_column.max() - pincp_column.min()) / 10
    # Create bins
    bins = np.arange(pincp_column.min(), pincp_column.max() + bin_size, bin_size)
    # Assign values to bins
    pincp_bins = pd.cut(pincp_column, bins=bins, labels=False)
    # Add new column
    ca_features_filt["PINCP_cat"] = pincp_bins

    ###### Create age bins
    # Assuming 'ca_features_filt' is your DataFrame
    age_column = ca_features_filt["AGEP"]
    # Calculate bin size
    bin_size = (age_column.max() - age_column.min()) / 5
    # Create bins
    bins = np.arange(age_column.min(), age_column.max() + bin_size, bin_size)
    # Assign values to bins
    age_bins = pd.cut(age_column, bins=bins, labels=False)
    # Add new column
    ca_features_filt["AGEP_cat"] = age_bins

    ###### Output label to int
    ca_labels_filt['PUBCOV'] = ca_labels_filt['PUBCOV'].astype(int)

    #print(ca_features_filt["AGEP_cat"])

    # Get the indices of rows with no NaNs in ca_features_filt
    valid_indices = ca_features_filt.dropna().index

    # Filter ca_labels_filt based on valid indices
    ca_features_filt = ca_features_filt.dropna()
    ca_labels_filt = ca_labels_filt.loc[valid_indices]

    in_cols = ["AGEP_cat", "SEX", "SCHL", "MAR", "RAC1P", "DIS", "CIT", "MIG", "DEAR", "DEYE", "DREM", "PINCP_cat", "FER"]
    out_cols = ["PUBCOV"]
    x_train, x_val, y_train, y_val = train_test_split(ca_features_filt[in_cols].values, ca_labels_filt[out_cols].values, test_size  = 0.30)

    # Normalization
    scaler = StandardScaler()  
    # Fitting only on training data
    scaler.fit(x_train)  
    X_train = scaler.transform(x_train)  
    # Applying same transformation to test data
    X_val = scaler.transform(x_val)
    
    file_path_raw = os.path.join(data_dir, 'val_data_raw_coverage.csv')
    file_path_scaled = os.path.join(data_dir, 'val_data_scaled_coverage.csv')

    data_combined_raw = np.concatenate((x_val, y_val), axis=1)

    # Convert the combined data to a DataFrame
    df_combined = pd.DataFrame(data_combined_raw)

    #x_val_columns = ['priors_count', 'score_code', 'age_code', 'gender_code', 'race_code', 'crime_code', 'charge_degree_code']
    #y_val_columns = ['two_year_recid']

    if not os.path.exists(file_path_raw):
        df_combined.to_csv(file_path_raw)
        print("File saved successfully.")
    else:
        print("File already exists. Not saving again.")


    data_combined_scaled = np.concatenate((X_val, y_val), axis=1)
    df_combined = pd.DataFrame(data_combined_scaled)
    if not os.path.exists(file_path_scaled):
        df_combined.to_csv(file_path_scaled)
        print("File saved successfully.")
    else:
        print("File already exists. Not saving again.")

    return x_train, X_train, y_train, X_val, y_val