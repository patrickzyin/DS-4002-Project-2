import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import r2_score, mean_absolute_error

#Load CSV
df = pd.read_csv("Project2-Data.csv")

#Strip % signs and convert to float
for col in df.columns:
    if col != "Month" and df[col].dtype == object:
        df[col] = df[col].str.replace("%", "").astype(float)

#Drop highly collinear features (|r| >= 0.85)
cols_to_drop = [
    "Food at home",
    "Food away from home",
    "Energy",
    "Services less energy services",
    "All items less food and energy",
    "All items"
]
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

#Separate target and Lasso-selected exogenous features
ts = df["UMCSENT"].reset_index(drop=True)
lasso_selected = [
    "Food",
    "Shelter",
    "Apparel",
    "Natural gas (piped)",
    "Education and communication"
]
exog = df[[c for c in lasso_selected if c in df.columns]].reset_index(drop=True)

#Standardize exogenous features
exog_scaled = (exog - exog.mean()) / exog.std()

#80/20 time-ordered split
split = int(len(ts) * 0.8)
train_y = ts.iloc[:split]
test_y  = ts.iloc[split:]
train_x = exog_scaled.iloc[:split]
test_x  = exog_scaled.iloc[split:]

print(f"Total rows: {len(ts)}")
print(f"Train size: {len(train_y)}, Test size: {len(test_y)}")
print(f"Train mean: {train_y.mean():.2f}")
print(f"Test mean:  {test_y.mean():.2f}")
print(f"Exogenous features: {list(exog.columns)}")

#Stationarity check
adf_raw = adfuller(train_y)
adf_diff = adfuller(train_y.diff().dropna())
print(f"\nADF Statistic (raw):         {adf_raw[0]:.4f}, p={adf_raw[1]:.4f}")
print(f"ADF Statistic (differenced): {adf_diff[0]:.4f}, p={adf_diff[1]:.4f}")

#Rolling window forecast
print("\nRunning rolling window forecast (this may take a minute)...")
history_y = list(train_y)
history_x = list(train_x.values)
predictions_list = []

for t in range(len(test_y)):
    model = ARIMA(history_y, exog=history_x, order=(1, 1, 1))
    model_fit = model.fit()
    yhat = model_fit.forecast(steps=1, exog=[test_x.iloc[t].values])[0]
    predictions_list.append(yhat)
    history_y.append(test_y.iloc[t])
    history_x.append(test_x.iloc[t].values)

predictions = np.array(predictions_list)

r2  = r2_score(test_y, predictions)
mae = mean_absolute_error(test_y, predictions)

print(model_fit.summary())

print("=" * 45)
print("            ARIMAX Results")
print("=" * 45)
print(f"  R-squared : {r2:.4f}")
print(f"  MAE       : {mae:.4f}")
print("=" * 45)

#Save results
with open("arimax_results.txt", "w", encoding="utf-8") as f:
    f.write("=" * 45 + "\n")
    f.write("            ARIMAX Results (Rolling Window)\n")
    f.write("=" * 45 + "\n")
    f.write(f"  R-squared : {r2:.4f}\n")
    f.write(f"  MAE       : {mae:.4f}\n")
    f.write("=" * 45 + "\n\n")
    f.write(f"Exogenous features: {list(exog.columns)}\n\n")
    f.write(f"Train mean: {train_y.mean():.2f}\n")
    f.write(f"Test mean:  {test_y.mean():.2f}\n\n")
    f.write(f"ADF Statistic (raw):         {adf_raw[0]:.4f}, p={adf_raw[1]:.4f}\n")
    f.write(f"ADF Statistic (differenced): {adf_diff[0]:.4f}, p={adf_diff[1]:.4f}\n\n")
    f.write(model_fit.summary().as_text())

#Plot: Actual vs Predicted
plt.figure(figsize=(12, 4))
plt.plot(range(len(test_y)), test_y.values, label="Actual", linewidth=2)
plt.plot(range(len(test_y)), predictions, label="Predicted", linewidth=2, linestyle="--")
plt.title("ARIMAX: Actual vs Predicted Consumer Sentiment (Rolling Window)")
plt.xlabel("Month (test set index)")
plt.ylabel("UMCSENT")
plt.legend()
plt.tight_layout()
plt.savefig("arimax_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.show()

#Plot: Full series with forecast overlaid
plt.figure(figsize=(14, 4))
plt.plot(range(len(train_y)), train_y.values, label="Train", linewidth=2)
plt.plot(range(len(train_y), len(train_y) + len(test_y)), test_y.values, label="Actual (Test)", linewidth=2)
plt.plot(range(len(train_y), len(train_y) + len(test_y)), predictions, label="ARIMAX Forecast", linewidth=2, linestyle="--")
plt.title("ARIMAX: Full Series with Forecast")
plt.xlabel("Month index")
plt.ylabel("UMCSENT")
plt.legend()
plt.tight_layout()
plt.savefig("arimax_full_series.png", dpi=150, bbox_inches="tight")
plt.show()