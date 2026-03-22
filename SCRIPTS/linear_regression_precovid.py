import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

#Output folder
output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "OUTPUT", "linear_regression_precovid")
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

#Separate features and target
X = df.drop(columns=["UMCSENT"])
y = df["UMCSENT"]

#Stationarity check
adf_raw = adfuller(y)
adf_diff = adfuller(y.diff().dropna())
print(f"ADF Statistic (raw):         {adf_raw[0]:.4f}, p={adf_raw[1]:.4f}")
print(f"ADF Statistic (differenced): {adf_diff[0]:.4f}, p={adf_diff[1]:.4f}")

#Standardize features
X_scaled = (X - X.mean()) / X.std()

#Time ordered 80/20 split
split = int(len(df) * 0.8)
X_train, X_test = X_scaled.iloc[:split], X_scaled.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

#Fit model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2_ordered  = r2_score(y_test, y_pred)
mae_ordered = mean_absolute_error(y_test, y_pred)

#Shuffled split for diagnostic
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model_s = LinearRegression()
model_s.fit(X_train_s, y_train_s)
y_pred_s = model_s.predict(X_test_s)
r2_shuffled  = r2_score(y_test_s, y_pred_s)
mae_shuffled = mean_absolute_error(y_test_s, y_pred_s)

#Print results
print(f"\nTotal rows (pre-COVID): {len(df)}")
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
print(f"Train y mean: {y_train.mean():.2f}, Test y mean: {y_test.mean():.2f}")

print("=" * 20)
print("Linear Regression (Pre-COVID) — Time-Ordered")
print("=" * 20)
print(f"R-squared : {r2_ordered:.4f}  (goal: >= 0.70)")
print(f"MAE       : {mae_ordered:.4f}")
print("=" * 20)
print("=" * 20)
print("Linear Regression (Pre-COVID) — Shuffled")
print("=" * 20)
print(f"R-squared : {r2_shuffled:.4f}")
print(f"MAE       : {mae_shuffled:.4f}")
print("=" * 20)

#T-test coefficients
X_train_sm = sm.add_constant(X_train)
ols = sm.OLS(y_train, X_train_sm).fit()
print("\nCoefficient Summary (α = 0.05):")
print(ols.summary2().tables[1][["Coef.", "Std.Err.", "t", "P>|t|"]].to_string())

#Save results
with open(os.path.join(output_dir, "linear_regression_precovid_results.txt"), "w", encoding="utf-8") as f:
    f.write("Linear Regression — Pre-COVID (1980–March 2020)\n")
    f.write("=" * 20 + "\n")
    f.write("Time-Ordered Split\n")
    f.write("=" * 20 + "\n")
    f.write(f"R-squared : {r2_ordered:.4f}\n")
    f.write(f"MAE       : {mae_ordered:.4f}\n")
    f.write(f"Train y mean: {y_train.mean():.2f}, Test y mean: {y_test.mean():.2f}\n\n")
    f.write(f"ADF (raw): {adf_raw[0]:.4f}, p={adf_raw[1]:.4f}\n")
    f.write(f"ADF (differenced): {adf_diff[0]:.4f}, p={adf_diff[1]:.4f}\n\n")
    f.write("=" * 20 + "\n")
    f.write("Shuffled Split (diagnostic)\n")
    f.write("=" * 20 + "\n")
    f.write(f"R-squared : {r2_shuffled:.4f}\n")
    f.write(f"MAE       : {mae_shuffled:.4f}\n\n")
    f.write("Coefficient Summary (α = 0.05):\n")
    f.write(ols.summary2().tables[1][["Coef.", "Std.Err.", "t", "P>|t|"]].to_string())

#Actual vs predicted plot
plt.figure(figsize=(12, 4))
plt.plot(y_test.values, label="Actual", linewidth=2)
plt.plot(y_pred, label="Predicted", linewidth=2, linestyle="--")
plt.title("Linear Regression (Pre-COVID): Actual vs Predicted (Time-Ordered)")
plt.xlabel("Month (test set index)")
plt.ylabel("UMCSENT")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "linear_precovid_actual_vs_predicted.png"), dpi=150, bbox_inches="tight")
plt.show()

#Shuffled plot
plt.figure(figsize=(12, 4))
plt.plot(y_test_s.values, label="Actual", linewidth=2)
plt.plot(y_pred_s, label="Predicted", linewidth=2, linestyle="--")
plt.title("Linear Regression (Pre-COVID): Actual vs Predicted (Shuffled)")
plt.xlabel("Observation index")
plt.ylabel("UMCSENT")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "linear_precovid_actual_vs_predicted_shuffled.png"), dpi=150, bbox_inches="tight")
plt.show()

#Coefficient plot
coef_df = pd.DataFrame({"Feature": X_train.columns, "Coefficient": model.coef_}).sort_values("Coefficient")
colors = ["crimson" if c < 0 else "steelblue" for c in coef_df["Coefficient"]]
plt.figure(figsize=(10, 5))
plt.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors)
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Standardized Coefficients – Linear Regression (Pre-COVID)")
plt.xlabel("Coefficient Value")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "linear_precovid_coefficients.png"), dpi=150, bbox_inches="tight")
plt.show()