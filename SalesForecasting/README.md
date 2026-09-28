# Sales Forecasting Project

## Goal
Predict future retail/e-commerce sales using historical time-series data.

## Features included
- Historical monthly sales
- Trend component
- Monthly seasonality
- Promotion indicator
- Holiday-season indicator
- Discount percentage
- Advertising spend

## Models
1. Naive seasonal baseline
2. ARIMA / SARIMAX with promotional regressors
3. Prophet (if installed)
4. Optional LSTM template

## Visualizations
The notebook/script creates:
- Historical sales trend
- Monthly seasonal pattern
- Promotion vs non-promotion sales
- Train/test actual vs forecast
- Future sales forecast

## Project structure
- `data/sales_data.csv` - sample dataset
- `notebooks/sales_forecasting.ipynb` - complete notebook
- `src/forecast.py` - reusable forecasting script
- `requirements.txt` - Python packages
- `outputs/` - generated charts/forecast files

## Run
```bash
pip install -r requirements.txt
python src/forecast.py
```

For the notebook:
```bash
jupyter notebook notebooks/sales_forecasting.ipynb
```

## Notes
Prophet and TensorFlow are optional because installation can be platform-dependent. The main executable workflow uses pandas, matplotlib, scikit-learn and statsmodels.
