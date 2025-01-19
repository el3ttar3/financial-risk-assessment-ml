import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def evaluate_model(y_true, y_pred, model_name):
    """
    Evaluate model performance using multiple metrics
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        'Model': model_name,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2
    }

def compare_models(models_results):
    """
    Compare multiple models based on their performance metrics
    """
    best_mse = float('inf')
    best_r2 = float('-inf')
    best_model_mse = None
    best_model_r2 = None

    for result in models_results:
        if result['MSE'] < best_mse:
            best_mse = result['MSE']
            best_model_mse = result['Model']
        
        if result['R²'] > best_r2:
            best_r2 = result['R²']
            best_model_r2 = result['Model']

    return {
        'Best Model (MSE)': best_model_mse,
        'Best MSE': best_mse,
        'Best Model (R²)': best_model_r2,
        'Best R²': best_r2
    } 