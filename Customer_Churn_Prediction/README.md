# Customer Churn Prediction

## Project Goal
Predict which customers are likely to stop using a telecom service using machine learning classification models.

## Models
- Logistic Regression
- Random Forest
- XGBoost

## Workflow
1. Load and clean the customer dataset.
2. Perform Exploratory Data Analysis (EDA).
3. Encode categorical variables and split the data.
4. Train three classification models.
5. Evaluate Accuracy, Recall, ROC-AUC, Precision, F1-score and confusion matrices.
6. Save the best model.
7. Use the Streamlit dashboard for interactive predictions.

## Dataset
`data/customer_churn.csv` is a self-contained telecom-style dataset created for this academic project. It contains 1,200 customer records and realistic churn-related features.

## Installation
```bash
pip install -r requirements.txt
```

## Train Models
```bash
python src/train.py
```

The trained model and evaluation summary are saved in `models/` and `reports/`.

## Run Streamlit App
```bash
streamlit run src/app.py
```

## Project Structure
```text
Customer_Churn_Prediction/
├── data/
│   └── customer_churn.csv
├── models/
├── notebooks/
├── reports/
├── src/
│   ├── eda.py
│   ├── train.py
│   └── app.py
├── requirements.txt
└── README.md
```

## Key Questions
- Which customer groups have higher churn?
- Does contract type affect churn?
- Does tenure affect churn?
- Which model performs best according to recall and ROC-AUC?
- Can the model identify customers who may churn?

## Academic Note
This project is intended for learning and portfolio/assignment purposes. The dataset is synthetic and should not be treated as real customer data.
