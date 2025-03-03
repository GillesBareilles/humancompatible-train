import pandas as pd
from IPython.display import Markdown, display
from sklearn.model_selection import train_test_split
import numpy as np
import os
import sys

from sklearn.preprocessing import StandardScaler 

RACE_IND = 4
SENSITIVE_CODE_0 = 0
SENSITIVE_CODE_1 = 1


def load_compas() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'raw_data', 'compas'))
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

    x_train, x_val, y_train, y_val = train_test_split(in_df.values, out_df.values, test_size  = 0.30)

    # Normalization
    scaler = StandardScaler()  

    # Fitting only on training data
    scaler.fit(x_train)  
    X_train = scaler.transform(x_train)  

    # Applying same transformation to test data
    X_val = scaler.transform(x_val)

    # Concatenate x_val and y_val along the columns

    data_combined_raw = np.concatenate((x_val, y_val), axis=1)

    # Convert the combined data to a DataFrame
    df_combined = pd.DataFrame(data_combined_raw)

    #x_val_columns = ['priors_count', 'score_code', 'age_code', 'gender_code', 'race_code', 'crime_code', 'charge_degree_code']
    #y_val_columns = ['two_year_recid']

    file_path_raw = os.path.join(data_dir, 'val_data_raw_compas.csv')
    file_path_scaled = os.path.join(data_dir, 'val_data_scaled_compas.csv')

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