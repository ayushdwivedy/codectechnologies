# Customer Churn Prediction — Project Report

## Abstract
Customer churn prediction helps service providers identify customers who may discontinue their services. This project applies exploratory data analysis and supervised machine learning to a telecom-style customer dataset. Logistic Regression, Random Forest and XGBoost are trained and compared using accuracy, recall and ROC-AUC along with precision and F1-score.

## Problem Statement
Companies lose revenue when customers discontinue their services. A predictive model can identify high-risk customers early so that the business can study retention strategies.

## Objectives
1. Analyze customer attributes related to churn.
2. Perform data cleaning and exploratory analysis.
3. Build classification models.
4. Compare model performance.
5. Provide an interactive prediction interface.

## Methodology
Data collection → Cleaning → EDA → Preprocessing → Train/Test Split → Model Training → Evaluation → Prediction App.

## Models
### Logistic Regression
A linear classification model that estimates the probability of churn.

### Random Forest
An ensemble of decision trees that can capture non-linear relationships.

### XGBoost
A gradient-boosting model that builds trees sequentially to improve predictive performance.

## Evaluation Metrics
- Accuracy: overall percentage of correct predictions.
- Recall: proportion of actual churners correctly identified.
- ROC-AUC: ability to distinguish churn and non-churn customers across thresholds.

## Results
Run `python src/train.py` to generate `reports/model_results.csv`. The script automatically saves the model with the highest ROC-AUC as `models/best_model.joblib`.

## Conclusion
The project demonstrates a complete end-to-end customer churn prediction pipeline from EDA to model deployment. The saved Streamlit application allows a user to enter customer information and receive a churn probability.
