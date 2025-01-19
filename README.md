# Financial Risk Assessment ML Project

## Overview
This project implements a machine learning solution for financial risk assessment using various regression models. The project is divided into two phases:
1. Phase 1: Initial model development and evaluation
2. Phase 2: Advanced model implementation with feature engineering

## Data
The project uses two main datasets:
- `Loan.csv`: Contains loan application data with features like credit score, income, etc.
- `Regression_test_file.csv`: Test dataset for model evaluation

To get started with this project:
1. Create a `data/` directory in the project root
2. Download the required datasets from [This Drive](https://drive.google.com/drive/folders/1_UcPiHOdTqc3QjCf7ccm5PSghmUUy8_5?usp=share_link)
3. Place the CSV files in the `data/` directory

## Project Structure
```
├── data/
│   ├── Loan.csv
│   └── Regression_test_file.csv
├── notebooks/
│   ├── phase1_initial_models.ipynb
│   └── phase2_advanced_models.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   └── model_evaluation.py
├── requirements.txt
└── README.md
```

## Features
- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA) with visualizations
- Feature engineering and transformation
- Implementation of multiple regression models:
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - K-Nearest Neighbors
  - Support Vector Regression (SVR)
  - Random Forest
  - Gradient Boosting
- Model evaluation and comparison
- Cross-validation and hyperparameter tuning

## Key Findings
- Comprehensive correlation analysis between financial variables
- Feature importance analysis for risk assessment
- Model performance comparison using metrics:
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - R-squared (R²)
  - Mean Absolute Error (MAE)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/financial-risk-assessment-ml.git
cd financial-risk-assessment-ml
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Data Preparation:
   - Place your loan dataset in the `data/` directory
   - Run data preprocessing scripts

2. Model Training:
   - Execute notebooks in sequence:
     1. `financial_risk_assessment_ml_project.py`
     2. `financial_risk_assessment(2)_ml_project.py`

3. Model Evaluation:
   - Review model performance metrics
   - Generate predictions on test data

## Model Performance
The project implements and compares several regression models:
- Random Forest Regressor
- Gradient Boosting Regressor
- SVR (Linear and RBF kernels)
- Linear Regression models

Best performing models based on R² score and MSE are documented in the notebooks.

## Dependencies
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- xgboost

## Contributing
Feel free to fork the repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/) 