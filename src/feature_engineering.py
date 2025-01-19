import numpy as np
from sklearn.preprocessing import PolynomialFeatures

def create_financial_ratios(df):
    """
    Create financial ratio features
    """
    df['Income_to_Loan_Ratio'] = df['AnnualIncome'] / df['LoanAmount']
    df['Debt_to_Assets_Ratio'] = df['TotalLiabilities'] / df['TotalAssets']
    df['Monthly_Income_to_Payment_Ratio'] = df['MonthlyIncome'] / df['MonthlyLoanPayment']
    return df

def create_polynomial_features(df):
    """
    Create polynomial features for credit score and debt-to-income ratio
    """
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_features = poly.fit_transform(df[['CreditScore', 'DebtToIncomeRatio']])
    df['CreditScore_Squared'] = poly_features[:, -2]
    df['DTI_Squared'] = poly_features[:, -1]
    return df

def engineer_features(df):
    """
    Apply all feature engineering steps
    """
    df = create_financial_ratios(df)
    df = create_polynomial_features(df)
    return df 