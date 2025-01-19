# -*- coding: utf-8 -*-
"""Financial_Risk_Assessment_ML-Project.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ttOIv_TBHiJ_r4nRfF3cq8l_Nr8CKySQ
"""

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, recall_score, roc_curve, auc, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Lasso

df = pd.read_csv("/content/Loan.csv")
df.head(20)

df.isnull().sum()

df.shape

duplicated = df[df.duplicated()]
duplicated

df.info()

df.describe()

df.drop('ApplicationDate', axis=1, errors='ignore', inplace=True)
df.head()

numerical_columns = df.select_dtypes(include=['number'])

plt.figure(figsize=(12, 8))
sns.boxplot(data=numerical_columns)
plt.title("Box Plot for Numerical Columns")
plt.xticks(rotation=45, ha="right")
plt.show()

numeric_features = df.select_dtypes(include=["number"])
numeric_features.columns

categorical_features = df.select_dtypes(include=["object"])
categorical_features.columns

categorical_features.head()

numeric_features.hist(bins=50, figsize=(20,15))
plt.show()

corr_matrix = numeric_features.corr()
corr_matrix["RiskScore"].sort_values(ascending=False)

#Visualize correlation
plt.subplots(figsize=(20,15))
ax = sns.heatmap(
    corr_matrix,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)

plt.figure(figsize=(20,10))

sns.heatmap(numeric_features.corr(),cmap='BrBG',fmt='.2f',
            linewidths=2,annot=True)

def box_plot(x, y, **kwargs):
    sns.boxplot(x=x, y=y)
    x=plt.xticks(rotation=90)

#a box plot to show how the output changes with categorical features
fi = pd.melt(df, id_vars=['Age'], value_vars = categorical_features)
ga = sns.FacetGrid(fi, col="variable",  col_wrap=2, sharex=False, sharey=False)
ga = ga.map(box_plot, "value", "Age")

plt.figure(figsize=(10, 6))

# Violin plot to show distribution of RiskScore for each LoanApproved value
sns.violinplot(x='LoanApproved', y='RiskScore', data=df)

# Adding titles and labels
plt.title('Relationship between Loan Approval and Risk Score', fontsize=16)
plt.xlabel('Loan Approved', fontsize=14)
plt.ylabel('Risk Score', fontsize=14)

# Show plot
plt.show()

plt.figure(figsize=(10, 6))

# Violin plot to show distribution of Age for each LoanApproved value
sns.violinplot(x='LoanApproved', y='Age', data=df)

# Adding titles and labels
plt.title('Relationship between Loan Approval and Age', fontsize=16)
plt.xlabel('Loan Approved', fontsize=14)
plt.ylabel('Age', fontsize=14)

# Show plot
plt.show()

plt.figure(figsize=(10, 6))

# Scatterplot to show the relationship
sns.scatterplot(x='LoanApproved', y='RiskScore', data=df, hue='LoanApproved', palette='coolwarm', s=100, alpha=0.6)

# Adding titles and labels
plt.title('Scatter Plot of Loan Approval vs. Risk Score', fontsize=16)
plt.xlabel('Loan Approved', fontsize=14)
plt.ylabel('Risk Score', fontsize=14)

# Show plot
plt.show()

# Initialize LabelEncoder
le = LabelEncoder()
df['EducationLevel'] = le.fit_transform(df['EducationLevel'])
df['MaritalStatus'] = le.fit_transform(df['MaritalStatus'])
df['EmploymentStatus'] = le.fit_transform(df['EmploymentStatus'])
df.head()

# One-hot encoding using pandas
df_encoded = pd.get_dummies(df, columns=['HomeOwnershipStatus', 'LoanPurpose'], drop_first=True)
df_encoded.head()

df_encoded.info()

# List of boolean columns to convert
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

df_encoded.info()

def log_transform(df_encoded, column):
    df_encoded[column] = np.log1p(df_encoded[column])  # log1p(x) is equivalent to log(1 + x), it helps avoid log(0)
    return df_encoded

columns_to_transform = ['NetWorth', 'TotalAssets', 'AnnualIncome', 'TotalLiabilities',
                        'SavingsAccountBalance', 'LoanAmount', 'CheckingAccountBalance'
                        ]

for col in columns_to_transform:
    df_encoded = log_transform(df_encoded, col)

print(df['NetWorth'].max())
print(df['NetWorth'].min())

print(df_encoded['NetWorth'].max())
print(df_encoded['NetWorth'].min())

numerical_columns_2 = df_encoded.select_dtypes(include=['number'])

plt.figure(figsize=(12, 8))
sns.boxplot(data=numerical_columns_2)
plt.title("Box Plot for Numerical Columns")
plt.xticks(rotation=45, ha="right")
plt.show()

columns_to_transform2 = ['MonthlyDebtPayments', 'MonthlyIncome', 'MonthlyLoanPayment', 'CreditScore']

for col1 in columns_to_transform2:
    df_encoded = log_transform(df_encoded, col1)

print(df_encoded['MonthlyIncome'].max())
print(df_encoded['MonthlyIncome'].min())

print(df['MonthlyIncome'].max())
print(df['MonthlyIncome'].min())

numerical_columns_3 = df_encoded.select_dtypes(include=['number'])

plt.figure(figsize=(12, 8))
sns.boxplot(data=numerical_columns_3)
plt.title("Box Plot for Numerical Columns")
plt.xticks(rotation=45, ha="right")
plt.show()

columns_to_transform3 = ['Age', 'Experience', 'PaymentHistory', 'RiskScore']

for col2 in columns_to_transform3:
    df_encoded = log_transform(df_encoded, col2)

numerical_columns_4 = df_encoded.select_dtypes(include=['number'])

plt.figure(figsize=(12, 8))
sns.boxplot(data=numerical_columns_4)
plt.title("Box Plot for Numerical Columns")
plt.xticks(rotation=45, ha="right")
plt.show()

df_encoded = log_transform(df_encoded, 'LoanDuration')

numerical_columns_5 = df_encoded.select_dtypes(include=['number'])

plt.figure(figsize=(12, 8))
sns.boxplot(data=numerical_columns_5)
plt.title("Box Plot for Numerical Columns")
plt.xticks(rotation=45, ha="right")
plt.show()

X = df_encoded.drop(columns=['LoanApproved'])
Y = df_encoded['LoanApproved']

X.head()

Y.head()

# X is the feature matrix and y is the target
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# fiting the model
model = LinearRegression()
model.fit(X_train, y_train)

# coefficients and feature importance
coefficients = pd.Series(model.coef_, index=X.columns).sort_values()

print("Feature Importance:")
print(coefficients)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# Train the model on scaled data
model.fit(X_scaled, y_train)

# Get the standardized coefficients
coefficients2 = pd.Series(model.coef_, index=X_train.columns).sort_values()

print("Standardized Feature Importance:")
print(coefficients2)

# Plot the feature importance (Non-standardized coefficients)
plt.figure(figsize=(10,6))
coefficients.plot(kind='barh', color='skyblue')
plt.title('Feature Importance (Non-Standardized Coefficients)')
plt.xlabel('Non-Standardized Coefficient Value')
plt.ylabel('Features')
plt.show()

# Plot the feature importance (standardized coefficients)
plt.figure(figsize=(10,6))
coefficients2.plot(kind='barh', color='skyblue')
plt.title('Feature Importance (Standardized Coefficients)')
plt.xlabel('Standardized Coefficient Value')
plt.ylabel('Features')
plt.show()

# Predicted on test data
y_pred = model.predict(X_test)

# Mean Squared Error (MSE)
mse_original = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error (MSE): {mse_original}')

# Root Mean Squared Error (RMSE)
rmse_original = np.sqrt(mse_original)
print(f'Root Mean Squared Error (RMSE): {rmse_original}')

# R-squared (R²)
r2_original = r2_score(y_test, y_pred)
print(f'R-squared (R²): {r2_original}')

def eval_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nClassification Report: \n",classification_report(y_test, y_pred))
    print(f"Recall: {recall_score(y_test, y_pred)}")
    print("AUC:", roc_auc_score(y_test, y_pred))
    print(f"F1-Score: {f1_score(y_test, y_pred)}")
    print('')

    # ROC Curve
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

# Logistic Regression Model
print("Logistic Regression Model:")
lr_model = LogisticRegression(random_state=42)
eval_model(lr_model, X_train, X_test, y_train, y_test)

model_results = {}

# Ridge Model
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)

y_pred = ridge_model.predict(X_test)

mse_ridg = mean_squared_error(y_test, y_pred)
r2_ridg = r2_score(y_test, y_pred)

rmse_original_ridg = np.sqrt(mse_ridg)
model_results['Ridge Regression'] = {'MSE': mse_ridg, 'R²': r2_ridg, 'RMSE': rmse_original_ridg}

print(f"Mean Squared Error (MSE): {mse_ridg}")
print(f'Root Mean Squared Error (RMSE): {rmse_original_ridg}')
print(f"R-squared (R²): {r2_ridg}")

# KNeighborsReg
KNR = KNeighborsRegressor(n_neighbors=3)
KNR.fit(X_train, y_train)
y_pred = KNR.predict(X_test)

mse_KNR = mean_squared_error(y_test, y_pred)
r2_KNR = r2_score(y_test, y_pred)

rmse_original_KNR = np.sqrt(mse_KNR)
model_results['KNeighbors Regression'] = {'MSE': mse_KNR, 'R²': r2_KNR, 'RMSE': rmse_original_KNR}

print(f"Mean Squared Error (MSE): {mse_KNR}")
print(f'Root Mean Squared Error (RMSE): {rmse_original_KNR}')
print(f"R-squared (R²): {r2_KNR}")

# Log-Reg
LOG_REG = LogisticRegression(max_iter=200)
LOG_REG.fit(X_train, y_train)
y_pred = LOG_REG.predict(X_test)

mse_log = mean_squared_error(y_test, y_pred)
r2_log = r2_score(y_test, y_pred)

rmse_original_log = np.sqrt(mse_log)
model_results['Logistic Regression'] = {'MSE': mse_log, 'R²': r2_log, 'RMSE': rmse_original_log}

print(f"Mean Squared Error (MSE): {mse_log}")
print(f'Root Mean Squared Error (RMSE): {rmse_original_log}')
print(f"R-squared (R²): {r2_log}")

# Linear Regression
linReg = LinearRegression()
linReg.fit(X_train, y_train)
y_pred = linReg.predict(X_test)

mse_lin = mean_squared_error(y_test, y_pred)
r2_lin = r2_score(y_test, y_pred)

rmse_original_lin = np.sqrt(mse_lin)
model_results['Linear Regression'] = {'MSE': mse_lin, 'R²': r2_lin, 'RMSE': rmse_original_lin}

print(f"Mean Squared Error (MSE): {mse_lin}")
print(f'Root Mean Squared Error (RMSE): {rmse_original_lin}')
print(f"R-squared (R²): {r2_lin}")

# Lasso Regression
lasso_reg = Lasso(alpha=1.0)
lasso_reg.fit(X_train, y_train)
y_pred_lasso = lasso_reg.predict(X_test)

mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred)

rmse_original_lasso = np.sqrt(mse_lasso)

model_results['Lasso Regression'] = {'MSE': mse_lasso, 'R²': r2_lasso, 'RMSE': rmse_original_lasso}

print(f"Mean Squared Error (MSE): {mse_lasso}")
print(f'Root Mean Squared Error (RMSE): {rmse_original_lasso}')
print(f"R-squared (R²): {r2_lasso}")

results_df = pd.DataFrame(model_results).T  # better formatting
results_df

# Plot a bar chart to compare the models based on MSE
plt.figure(figsize=(10, 6))
plt.bar(results_df.index, results_df['MSE'], color=['blue', 'green', 'red', 'purple'])
plt.xlabel('Model')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Comparison of MSE for Different Regression Models')
plt.show()

# Plot a bar chart to compare the models based on R² score
plt.figure(figsize=(10, 6))
plt.bar(results_df.index, results_df['R²'], color=['blue', 'green', 'red', 'purple'])
plt.xlabel('Model')
plt.ylabel('R² Score')
plt.title('Comparison of R² Score for Different Regression Models')
plt.show()

# Plot a bar chart to compare the models based on RMSE score
plt.figure(figsize=(10, 6))
plt.bar(results_df.index, results_df['RMSE'], color=['blue', 'green', 'red', 'purple'])
plt.xlabel('Model')
plt.ylabel('RMSE Score')
plt.title('Comparison of RMSE Score for Different Regression Models')
plt.show()

best_model_mse = results_df['MSE'].idxmin()
best_model_r2 = results_df['R²'].idxmax()
print(f"Best model based on MSE: {best_model_mse}")
print(f"Best model based on R² Score: {best_model_r2}")

# R² in descending order (best to worst)
sorted_results = results_df.sort_values(by='R²', ascending=False)

print(sorted_results)

# the second best model based on R²
second_best_model = sorted_results.iloc[1]

print("\nSecond Best Model Based on R²:")
print(second_best_model)