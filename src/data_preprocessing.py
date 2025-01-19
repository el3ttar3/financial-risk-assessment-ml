import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_and_clean_data(file_path):
    """
    Load and perform initial data cleaning
    """
    df = pd.read_csv(file_path)
    df.drop('ApplicationDate', axis=1, errors='ignore', inplace=True)
    return df

def encode_categorical_features(df):
    """
    Encode categorical features using LabelEncoder and one-hot encoding
    """
    le = LabelEncoder()
    df['EducationLevel'] = le.fit_transform(df['EducationLevel'])
    df['MaritalStatus'] = le.fit_transform(df['MaritalStatus'])
    df['EmploymentStatus'] = le.fit_transform(df['EmploymentStatus'])
    
    # One-hot encoding
    df_encoded = pd.get_dummies(df, columns=['HomeOwnershipStatus', 'LoanPurpose'], drop_first=True)
    
    # Convert boolean columns to int
    bool_columns = [
        'HomeOwnershipStatus_Other',
        'HomeOwnershipStatus_Own',
        'HomeOwnershipStatus_Rent',
        'LoanPurpose_Debt Consolidation',
        'LoanPurpose_Education',
        'LoanPurpose_Home',
        'LoanPurpose_Other'
    ]
    df_encoded[bool_columns] = df_encoded[bool_columns].astype(int)
    
    return df_encoded

def log_transform_features(df, columns=None):
    """
    Apply log transformation to specified columns
    """
    if columns is None:
        columns = df.columns
    
    for col in columns:
        df[col] = np.log1p(df[col])
    
    return df 