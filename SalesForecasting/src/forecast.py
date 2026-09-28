import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sales_data.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA, parse_dates=["date"]).set_index("date").asfreq("MS")
df["month"] = df.index.month

# -------------------- EDA --------------------
plt.figure(figsize=(12,5))
plt.plot(df.index, df["sales"], label="Actual sales")
plt.title("Historical Monthly Sales")
plt.xlabel("Date"); plt.ylabel("Sales")
plt.grid(alpha=.25); plt.legend()
plt.tight_layout(); plt.savefig(OUT/"historical_sales.png", dpi=150); plt.close()

monthly = df.groupby("month")["sales"].mean()
plt.figure(figsize=(10,4))
plt.plot(monthly.index, monthly.values, marker="o")
plt.title("Average Sales by Month")
plt.xlabel("Month"); plt.ylabel("Average Sales")
plt.xticks(range(1,13)); plt.grid(alpha=.25)
plt.tight_layout(); plt.savefig(OUT/"seasonality.png", dpi=150); plt.close()

# -------------------- Train/Test --------------------
test_size = 12
train = df.iloc[:-test_size].copy()
test = df.iloc[-test_size:].copy()

exog_cols = ["promotion", "holiday_season", "discount_pct", "ad_spend"]
model = SARIMAX(
    train["sales"],
    exog=train[exog_cols],
    order=(1,1,1),
    seasonal_order=(1,1,1,12),
    enforce_stationarity=False,
    enforce_invertibility=False
)
fit = model.fit(disp=False)

pred = fit.get_forecast(steps=len(test), exog=test[exog_cols]).predicted_mean
mae = mean_absolute_error(test["sales"], pred)
rmse = mean_squared_error(test["sales"], pred) ** 0.5

comparison = pd.DataFrame({"actual": test["sales"], "forecast": pred})
comparison.to_csv(OUT/"test_forecast.csv")

plt.figure(figsize=(12,5))
plt.plot(train.index, train["sales"], label="Train")
plt.plot(test.index, test["sales"], label="Actual")
plt.plot(test.index, pred, label="SARIMAX Forecast")
plt.axvline(test.index[0], linestyle="--", alpha=.6)
plt.title(f"Actual vs Forecast | MAE={mae:,.0f} | RMSE={rmse:,.0f}")
plt.xlabel("Date"); plt.ylabel("Sales")
plt.grid(alpha=.25); plt.legend()
plt.tight_layout(); plt.savefig(OUT/"actual_vs_forecast.png", dpi=150); plt.close()

# -------------------- Future forecast --------------------
future_dates = pd.date_range(df.index[-1] + pd.offsets.MonthBegin(1), periods=12, freq="MS")
future = pd.DataFrame(index=future_dates)
future["promotion"] = future.index.month.isin([8,11,12]).astype(int)
future["holiday_season"] = future.index.month.isin([10,11,12]).astype(int)
future["discount_pct"] = np.where(
    future["promotion"], 20, 5
)
# Simple planned advertising assumption for demonstration
future["ad_spend"] = np.where(future["promotion"], 23000, 12000)

future_pred = fit.get_forecast(steps=12, exog=future[exog_cols])
future["forecast_sales"] = future_pred.predicted_mean.values
future["lower_95"] = future_pred.conf_int().iloc[:,0].values
future["upper_95"] = future_pred.conf_int().iloc[:,1].values
future.to_csv(OUT/"future_sales_forecast.csv")

plt.figure(figsize=(12,5))
plt.plot(df.index, df["sales"], label="Historical sales")
plt.plot(future.index, future["forecast_sales"], label="Future forecast")
plt.fill_between(future.index, future["lower_95"], future["upper_95"], alpha=.2, label="95% interval")
plt.title("Next 12 Months Sales Forecast")
plt.xlabel("Date"); plt.ylabel("Sales")
plt.grid(alpha=.25); plt.legend()
plt.tight_layout(); plt.savefig(OUT/"future_forecast.png", dpi=150); plt.close()

print("Sales Forecasting Project completed.")
print(f"MAE: {mae:,.2f}")
print(f"RMSE: {rmse:,.2f}")
print("\nNext 12 months:")
print(future[["forecast_sales","lower_95","upper_95"]].round(2))
