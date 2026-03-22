import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

#Load CSV
df = pd.read_csv("Project2-Data.csv")
df = df.drop(columns=["Month"])

#Strip % signs and convert to float
for col in df.columns:
    if df[col].dtype == object:
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

#Separate features and target
X = df.drop(columns=["UMCSENT"])
y = df["UMCSENT"]

#Stationarity check on target
adf_raw = adfuller(y)
adf_diff = adfuller(y.diff().dropna())
print(f"ADF Statistic (raw):         {adf_raw[0]:.4f}, p={adf_raw[1]:.4f}")
print(f"ADF Statistic (differenced): {adf_diff[0]:.4f}, p={adf_diff[1]:.4f}")

#Standardize features (mean=0, std=1)
X_scaled = (X - X.mean()) / X.std()

# RUN 1: Time-ordered split (no shuffle)
split = int(len(df) * 0.8)
X_train, X_test = X_scaled.iloc[:split], X_scaled.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2_ordered  = r2_score(y_test, y_pred)
mae_ordered = mean_absolute_error(y_test, y_pred)

# RUN 2: Shuffled split (diagnostic)
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
model_s = LinearRegression()
model_s.fit(X_train_s, y_train_s)
y_pred_s = model_s.predict(X_test_s)
r2_shuffled  = r2_score(y_test_s, y_pred_s)
mae_shuffled = mean_absolute_error(y_test_s, y_pred_s)

#Print results
print(f"Total rows: {len(df)}")
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
print(f"Train y mean (ordered): {y_train.mean():.2f}")
print(f"Test y mean (ordered):  {y_test.mean():.2f}")

print("=" * 45)
print("   Linear Regression — Time-Ordered Split")
print("=" * 45)
print(f"  R-squared : {r2_ordered:.4f}  (goal: >= 0.70)")
print(f"  MAE       : {mae_ordered:.4f}")
print("=" * 45)

print("=" * 45)
print("   Linear Regression — Shuffled Split (diagnostic)")
print("=" * 45)
print(f"  R-squared : {r2_shuffled:.4f}")
print(f"  MAE       : {mae_shuffled:.4f}")
print("=" * 45)

#T-test on time-ordered model
X_train_sm = sm.add_constant(X_train)
ols = sm.OLS(y_train, X_train_sm).fit()
print("\nCoefficient Summary (α = 0.05):")
print(ols.summary2().tables[1][["Coef.", "Std.Err.", "t", "P>|t|"]].to_string())

#Save results
with open("linear_regression_results.txt", "w", encoding="utf-8") as f:
    f.write("=" * 45 + "\n")
    f.write("   Linear Regression — Time-Ordered Split\n")
    f.write("=" * 45 + "\n")
    f.write(f"  R-squared : {r2_ordered:.4f}  (goal: >= 0.70)\n")
    f.write(f"  MAE       : {mae_ordered:.4f}\n")
    f.write("=" * 45 + "\n\n")
    f.write(f"Train y mean: {y_train.mean():.2f}\n")
    f.write(f"Test y mean:  {y_test.mean():.2f}\n\n")
    f.write(f"ADF Statistic (raw):         {adf_raw[0]:.4f}, p={adf_raw[1]:.4f}\n")
    f.write(f"ADF Statistic (differenced): {adf_diff[0]:.4f}, p={adf_diff[1]:.4f}\n\n")
    f.write("=" * 45 + "\n")
    f.write("   Linear Regression — Shuffled Split (diagnostic)\n")
    f.write("=" * 45 + "\n")
    f.write(f"  R-squared : {r2_shuffled:.4f}\n")
    f.write(f"  MAE       : {mae_shuffled:.4f}\n")
    f.write("=" * 45 + "\n\n")
    f.write("Coefficient Summary (α = 0.05):\n")
    f.write(ols.summary2().tables[1][["Coef.", "Std.Err.", "t", "P>|t|"]].to_string())

#Plot: Actual vs Predicted (time-ordered)
plt.figure(figsize=(12, 4))
plt.plot(y_test.values, label="Actual", linewidth=2)
plt.plot(y_pred, label="Predicted", linewidth=2, linestyle="--")
plt.title("Linear Regression: Actual vs Predicted Consumer Sentiment (Time-Ordered)")
plt.xlabel("Month (test set index)")
plt.ylabel("UMCSENT")
plt.legend()
plt.tight_layout()
plt.savefig("linear_regression_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.show()

#Plot: Actual vs Predicted (shuffled)
plt.figure(figsize=(12, 4))
plt.plot(y_test_s.values, label="Actual", linewidth=2)
plt.plot(y_pred_s, label="Predicted", linewidth=2, linestyle="--")
plt.title("Linear Regression: Actual vs Predicted Consumer Sentiment (Shuffled)")
plt.xlabel("Observation index")
plt.ylabel("UMCSENT")
plt.legend()
plt.tight_layout()
plt.savefig("linear_regression_actual_vs_predicted_shuffled.png", dpi=150, bbox_inches="tight")
plt.show()

#Plot: Coefficients (time-ordered model)
coef_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Coefficient": model.coef_
}).sort_values("Coefficient")

colors = ["crimson" if c < 0 else "steelblue" for c in coef_df["Coefficient"]]

plt.figure(figsize=(10, 5))
plt.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors)
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Standardized Coefficients – Linear Regression")
plt.xlabel("Coefficient Value")
plt.tight_layout()
plt.savefig("linear_regression_coefficients.png", dpi=150, bbox_inches="tight")
plt.show()