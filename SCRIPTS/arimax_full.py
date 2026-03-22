import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import r2_score, mean_absolute_error

output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "OUTPUT", "arimax_full")
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv("../DATA/Project2-Data.csv")
df = df.dropna(subset=["UMCSENT"])
df = df.dropna(how="all")
df["Year"] = pd.to_datetime(df["Year"])
df = df.drop(columns=["Year"])

cols_to_drop = [
    "Food at home", "Food away from home", "Energy",
    "Services Less Energy Services", "Core CPI", "Overall CPI"
]
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

ts = df["UMCSENT"].reset_index(drop=True)
lasso_selected = ["Food"]
exog = df[[c for c in lasso_selected if c in df.columns]].reset_index(drop=True)
exog_scaled = (exog - exog.mean()) / exog.std()

split = int(len(ts) * 0.8)
train_y = ts.iloc[:split]
test_y  = ts.iloc[split:]
train_x = exog_scaled.iloc[:split]
test_x  = exog_scaled.iloc[split:]

print(f"Total rows: {len(ts)}")
print(f"Train mean: {train_y.mean():.2f}, Test mean: {test_y.mean():.2f}")
print(f"Exogenous features: {list(exog.columns)}")

adf_raw = adfuller(train_y)
adf_diff = adfuller(train_y.diff().dropna())
print(f"ADF (raw): {adf_raw[0]:.4f}, p={adf_raw[1]:.4f}")
print(f"ADF (differenced): {adf_diff[0]:.4f}, p={adf_diff[1]:.4f}")

# --- Cold forecast ---
print("\nFitting ARIMAX model...")
model = ARIMA(train_y, exog=train_x, order=(1, 0, 1))
model_fit = model.fit()

forecast_result = model_fit.get_forecast(steps=len(test_y), exog=test_x.values)
forecast_mean = forecast_result.predicted_mean
forecast_ci = forecast_result.conf_int()

r2  = r2_score(test_y, forecast_mean)
mae = mean_absolute_error(test_y, forecast_mean)

print(model_fit.summary())
print("=" * 45)
print("      ARIMAX Results (Full — Cold Forecast)")
print("=" * 45)
print(f"  R-squared : {r2:.4f}")
print(f"  MAE       : {mae:.4f}")
print("=" * 45)

with open(os.path.join(output_dir, "arimax_full_results.txt"), "w", encoding="utf-8") as f:
    f.write("ARIMAX — Full Dataset (1980–2026, Cold Forecast)\n")
    f.write("=" * 45 + "\n")
    f.write(f"  R-squared : {r2:.4f}\n")
    f.write(f"  MAE       : {mae:.4f}\n")
    f.write("=" * 45 + "\n\n")
    f.write(f"Exogenous features: {list(exog.columns)}\n")
    f.write(f"Train mean: {train_y.mean():.2f}, Test mean: {test_y.mean():.2f}\n\n")
    f.write(f"ADF (raw): {adf_raw[0]:.4f}, p={adf_raw[1]:.4f}\n")
    f.write(f"ADF (differenced): {adf_diff[0]:.4f}, p={adf_diff[1]:.4f}\n\n")
    f.write(model_fit.summary().as_text())

# --- Plot: Cold forecast with confidence interval ---
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
plt.title("ARIMAX (Full): Cold Forecast with 95% Confidence Interval")
plt.xlabel("Month index")
plt.ylabel("UMCSENT")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "arimax_full_cold_forecast.png"), dpi=150, bbox_inches="tight")
plt.show()

# --- Plot: Test set only ---
plt.figure(figsize=(12, 4))
plt.plot(range(len(test_y)), test_y.values, label="Actual", linewidth=2, color="steelblue")
plt.plot(range(len(test_y)), forecast_mean.values, label="Forecast", linewidth=2, color="crimson", linestyle="--")
plt.fill_between(
    range(len(test_y)),
    forecast_ci.iloc[:, 0],
    forecast_ci.iloc[:, 1],
    color="pink", alpha=0.3, label="95% Confidence Interval"
)
plt.title("ARIMAX (Full): Forecast vs Actual (Test Set)")
plt.xlabel("Month (test set index)")
plt.ylabel("UMCSENT")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "arimax_full_actual_vs_predicted.png"), dpi=150, bbox_inches="tight")
plt.show()