import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best_model.joblib"

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="wide")
st.title("📊 Customer Churn Prediction")
st.write("Predict whether a telecom customer is likely to churn.")

if not MODEL_PATH.exists():
    st.warning("Model not found. Run `python src/train.py` first.")
    st.stop()

model = joblib.load(MODEL_PATH)

st.sidebar.header("Customer Details")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior = st.sidebar.selectbox("Senior Citizen", [0, 1])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
phone = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
payment = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
paperless = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
monthly = st.sidebar.number_input("Monthly Charges", min_value=18.0, max_value=160.0, value=70.0)
total = st.sidebar.number_input("Total Charges", min_value=0.0, max_value=12000.0, value=float(monthly*tenure))

customer = pd.DataFrame([{
    "gender": gender,
    "SeniorCitizen": senior,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone,
    "InternetService": internet,
    "Contract": contract,
    "PaymentMethod": payment,
    "PaperlessBilling": paperless,
    "MonthlyCharges": monthly,
    "TotalCharges": total
}])

if st.button("Predict Churn", type="primary"):
    probability = model.predict_proba(customer)[0,1]
    prediction = "Likely to Churn" if probability >= 0.5 else "Likely to Stay"

    c1, c2 = st.columns(2)
    c1.metric("Prediction", prediction)
    c2.metric("Churn Probability", f"{probability:.1%}")

    st.progress(float(probability))
    if probability >= 0.5:
        st.error("This customer has a higher predicted churn risk.")
    else:
        st.success("This customer has a lower predicted churn risk.")

st.subheader("About the Project")
st.markdown("""
- **Models:** Logistic Regression, Random Forest, XGBoost
- **Evaluation:** Accuracy, Recall, Precision, F1-score, ROC-AUC
- **Purpose:** Academic/portfolio demonstration of a customer churn classification workflow.
""")
