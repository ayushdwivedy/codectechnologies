# Customer Churn Prediction — Notebook Workflow

## 1. Import libraries
Use pandas, numpy, matplotlib, seaborn, sklearn and xgboost.

## 2. Load dataset
Read `../data/customer_churn.csv`.

## 3. Data cleaning
- Convert TotalCharges to numeric.
- Handle missing values.
- Remove customerID from model features.

## 4. EDA
Study:
- Overall churn distribution
- Churn by contract
- Tenure vs churn
- Monthly charges vs churn

## 5. Preprocessing
- Median imputation for numeric columns.
- Most-frequent imputation for categorical columns.
- One-hot encoding for categorical variables.
- Standardization for numeric variables.

## 6. Model training
Train:
1. Logistic Regression
2. Random Forest
3. XGBoost

## 7. Evaluation
Compare Accuracy, Recall, Precision, F1-score and ROC-AUC.

## 8. Deployment
Run:
`streamlit run src/app.py`
