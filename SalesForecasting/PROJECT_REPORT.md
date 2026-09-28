# Sales Forecasting — Project Report

## 1. Introduction
Sales forecasting uses historical sales observations to estimate future demand. This project uses monthly retail/e-commerce sales data and applies a seasonal ARIMA-family model with external business variables.

## 2. Objectives
- Analyze historical sales.
- Identify trend and seasonality.
- Include promotion and holiday effects.
- Include discount and advertising-spend information.
- Train a time-series forecasting model.
- Compare actual and predicted sales.
- Generate a 12-month future forecast.

## 3. Dataset
The included dataset contains monthly observations from January 2019 to December 2025. It contains sales, promotion, holiday-season flag, discount percentage and advertising spend.

## 4. Methodology
1. Load and clean time-series data.
2. Convert date into a monthly index.
3. Perform exploratory analysis.
4. Split the final 12 months as a test set.
5. Train SARIMAX with 12-month seasonality and external regressors.
6. Evaluate using MAE and RMSE.
7. Forecast the next 12 months.

## 5. Results
Run `python src/forecast.py` or the notebook to generate the current metric values and charts. The outputs are saved in the `outputs` folder.

## 6. Conclusion
The project demonstrates how time-series forecasting can combine historical patterns with business drivers such as promotions, holidays, discounts and advertising spend. Forecasts should be updated regularly as new actual sales arrive.

## 7. Future Scope
- Hyperparameter tuning.
- Prophet model comparison.
- LSTM/deep-learning model.
- Product/category-level forecasting.
- Real company sales data.
- Automated dashboard using Power BI or Streamlit.
