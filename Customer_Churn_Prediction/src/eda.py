import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(ROOT / "data" / "customer_churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

sns.set_theme(style="whitegrid")

plt.figure(figsize=(7,5))
sns.countplot(data=df, x="Churn")
plt.title("Customer Churn Distribution")
plt.tight_layout()
plt.savefig(OUT/"01_churn_distribution.png", dpi=150)
plt.close()

plt.figure(figsize=(8,5))
sns.countplot(data=df, x="Contract", hue="Churn")
plt.title("Churn by Contract Type")
plt.tight_layout()
plt.savefig(OUT/"02_churn_by_contract.png", dpi=150)
plt.close()

plt.figure(figsize=(8,5))
sns.boxplot(data=df, x="Churn", y="tenure")
plt.title("Tenure vs Churn")
plt.tight_layout()
plt.savefig(OUT/"03_tenure_vs_churn.png", dpi=150)
plt.close()

plt.figure(figsize=(8,5))
sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
plt.title("Monthly Charges vs Churn")
plt.tight_layout()
plt.savefig(OUT/"04_monthly_charges_vs_churn.png", dpi=150)
plt.close()

summary = df.groupby("Contract")["Churn"].apply(lambda x: (x=="Yes").mean()).round(3)
summary.to_csv(OUT/"contract_churn_rates.csv", header=["churn_rate"])

print("EDA completed. Charts saved to reports/.")
