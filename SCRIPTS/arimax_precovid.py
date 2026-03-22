import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import r2_score, mean_absolute_error

#Output folder
output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "OUTPUT", "arimax_precovid")
os.makedirs(output_dir, exist_ok=True)

#Load data
df = pd.read_csv("../DATA/Project2-Data.csv")
df = df.dropna(subset=["UMCSENT"])
df = df.dropna(how="all")
df["Year"] = pd.to_datetime(df["Year"])

#Filter to pre-COVID
df = df[df["Year"] < "2020-04-01"].copy()
df = df.drop(columns=["Year"])
print(f"Rows after pre-COVID filter: {len(df)}")

#Drop collinear features
cols_to_drop = [
    "Food at home", "Food away from home", "Energy",
    "Services Less Energy Services", "Core CPI", "Overall CPI"
]
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

#Separate target and Lasso selected exogenous features
ts = df["UMCSENT"].reset_index(drop=True)
lasso_selected = [
    "Food",
    "Medical Care services",
    "Gasoline (all types)",
    "Electricity",
    "Commodities Less Food and Energy Commodities",
    "Natural Gas",
    "Medical Care Commodities",
    "Shelter"
]
exog = df[[c for c in lasso_selected if c in df.columns]].reset_index(drop=True)
exog_scaled = (exog - exog.mean()) / exog.std()

#Time ordered 80/20 split
split = int(len(ts) * 0.8)
train_y = ts.iloc[:split]
test_y  = ts.iloc[split:]
train_x = exog_scaled.iloc[:split]
test_x  = exog_scaled.iloc[split:]

print(f"Total rows (pre-COVID): {len(ts)}")
print(f"Train mean: {train_y.mean():.2f}, Test mean: {test_y.mean():.2f}")
print(f"Exogenous features: {list(exog.columns)}")

#Stationarity check
adf_raw = adfuller(train_y)
adf_diff = adfuller(train_y.diff().dropna())
print(f"ADF (raw): {adf_raw[0]:.4f}, p={adf_raw[1]:.4f}")
print(f"ADF (differenced): {adf_diff[0]:.4f}, p={adf_diff[1]:.4f}")

#Fit ARIMAX and generate cold forecast
print("\nFitting ARIMAX model...")
model = ARIMA(train_y, exog=train_x, order=(1, 1, 1))
model_fit = model.fit()

forecast_result = model_fit.get_forecast(steps=len(test_y), exog=test_x.values)
forecast_mean = forecast_result.predicted_mean
forecast_ci = forecast_result.conf_int()

r2  = r2_score(test_y, forecast_mean)
mae = mean_absolute_error(test_y, forecast_mean)

print(model_fit.summary())
print("=" * 20)
print("ARIMAX Results (Pre-COVID — Cold Forecast)")
print("=" * 20)
print(f"R-squared : {r2:.4f}")
print(f"MAE       : {mae:.4f}")
print("=" * 20)

#Save results
with open(os.path.join(output_dir, "arimax_precovid_results.txt"), "w", encoding="utf-8") as f:
    f.write("ARIMAX — Pre-COVID (1980–March 2020, Cold Forecast)\n")
    f.write("=" * 20 + "\n")
    f.write(f"R-squared : {r2:.4f}\n")
    f.write(f"MAE       : {mae:.4f}\n")
    f.write("=" * 20 + "\n\n")
    f.write(f"Exogenous features: {list(exog.columns)}\n")
    f.write(f"Train mean: {train_y.mean():.2f}, Test mean: {test_y.mean():.2f}\n\n")
    f.write(f"ADF (raw): {adf_raw[0]:.4f}, p={adf_raw[1]:.4f}\n")
    f.write(f"ADF (differenced): {adf_diff[0]:.4f}, p={adf_diff[1]:.4f}\n\n")
    f.write(model_fit.summary().as_text())

#Full series with forecast and confidence interval
plt.figure(figsize=(14, 5))
plt.plot(range(len(train_y)), train_y.values, label="Training Data", linewidth=2, color="steelblue")
plt.plot(range(len(train_y), len(train_y) + len(test_y)), test_y.values, label="Actual (Test)", linewidth=2, color="steelblue", linestyle="--")
plt.plot(range(len(train_y), len(train_y) + len(test_y)), forecast_mean.values, label="Cold Forecast", linewidth=2, color="crimson")
plt.fill_between(
    range(len(train_y), len(train_y) + len(test_y)),
    forecast_ci.iloc[:, 0],
    forecast_ci.iloc[:, 1],
    color="pink", alpha=0.3, label="95% Confidence Interval"
)
plt.axvline(x=len(train_y), color="black", linewidth=1, linestyle=":", label="Train/Test Split")
plt.title("ARIMAX (Pre-COVID): Cold Forecast with 95% Confidence Interval")
plt.xlabel("Month index")
plt.ylabel("UMCSENT")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "arimax_precovid_cold_forecast.png"), dpi=150, bbox_inches="tight")
plt.show()

#Test set only view
plt.figure(figsize=(12, 4))
plt.plot(range(len(test_y)), test_y.values, label="Actual", linewidth=2, color="steelblue")
plt.plot(range(len(test_y)), forecast_mean.values, label="Forecast", linewidth=2, color="crimson", linestyle="--")
plt.fill_between(
    range(len(test_y)),
    forecast_ci.iloc[:, 0],
    forecast_ci.iloc[:, 1],
    color="pink", alpha=0.3, label="95% Confidence Interval"
)
plt.title("ARIMAX (Pre-COVID): Forecast vs Actual (Test Set)")
plt.xlabel("Month (test set index)")
plt.ylabel("UMCSENT")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "arimax_precovid_actual_vs_predicted.png"), dpi=150, bbox_inches="tight")
plt.show()