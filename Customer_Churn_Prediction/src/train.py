import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, precision_score, f1_score, confusion_matrix
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(ROOT / "data" / "customer_churn.csv")
df = df.drop(columns=["customerID"])
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

X = df.drop(columns=["Churn"])
y = (df["Churn"] == "Yes").astype(int)

cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer([
    ("num", numeric_pipe, num_cols),
    ("cat", categorical_pipe, cat_cols)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=12, random_state=42, class_weight="balanced"
    ),
    "XGBoost": XGBClassifier(
        n_estimators=250, max_depth=5, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        eval_metric="logloss", random_state=42
    )
}

results = []
pipelines = {}

for name, model in models.items():
    pipe = Pipeline([("prep", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "ROC_AUC": roc_auc_score(y_test, proba)
    })
    pipelines[name] = pipe

results_df = pd.DataFrame(results).sort_values("ROC_AUC", ascending=False)
results_df.to_csv(REPORT_DIR/"model_results.csv", index=False)

# Save the model with highest ROC-AUC.
best_name = results_df.iloc[0]["Model"]
joblib.dump(pipelines[best_name], MODEL_DIR/"best_model.joblib")
(MODEL_DIR/"best_model_name.txt").write_text(best_name)

# Save test predictions for analysis
test_out = X_test.copy()
test_out["ActualChurn"] = y_test.values
test_out["PredictedChurn"] = pipelines[best_name].predict(X_test)
test_out["ChurnProbability"] = pipelines[best_name].predict_proba(X_test)[:,1]
test_out.to_csv(REPORT_DIR/"test_predictions.csv", index=False)

print(results_df.to_string(index=False))
print(f"\nBest model: {best_name}")
