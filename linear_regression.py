import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import statsmodels.api as sm

# --- Load & Clean ---
df = pd.read_csv("Project2-Data.csv")
df = df.drop(columns=["Month"])

# Strip % signs and convert to float
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].str.replace("%", "").astype(float)

# --- Drop highly collinear features (|r| >= 0.85) ---
cols_to_drop = [
    "Food at home",
    "Food away from home",
    "Energy",
    "Services less energy services",
    "All items less food and energy",
    "All items"
]
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# --- Separate features and target ---
X = df.drop(columns=["UMCSENT"])
y = df["UMCSENT"]

# --- Standardize features (mean=0, std=1) ---
X_scaled = (X - X.mean()) / X.std()

# --- 80/20 time-ordered split (no shuffle) ---
split = int(len(df) * 0.8)
X_train, X_test = X_scaled.iloc[:split], X_scaled.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

print(f"Total rows: {len(df)}")
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
print(f"Train y mean: {y_train.mean():.2f}")
print(f"Test y mean: {y_test.mean():.2f}")

# --- Sklearn OLS (for predictions + metrics) ---
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2  = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("=" * 45)
print("        Linear Regression Results")
print("=" * 45)
print(f"  R-squared : {r2:.4f}  (goal: >= 0.70)")
print(f"  MAE       : {mae:.4f}")
print("=" * 45)

# --- Statsmodels OLS (for p-values / t-tests) ---
X_train_sm = sm.add_constant(X_train)
ols = sm.OLS(y_train, X_train_sm).fit()
print("\nCoefficient Summary (α = 0.05):")
print(ols.summary2().tables[1][["Coef.", "Std.Err.", "t", "P>|t|"]].to_string())

# --- Save results summary ---
with open("linear_regression_results.txt", "w", encoding="utf-8") as f:
    f.write("=" * 45 + "\n")
    f.write("        Linear Regression Results\n")
    f.write("=" * 45 + "\n")
    f.write(f"  R-squared : {r2:.4f}  (goal: >= 0.70)\n")
    f.write(f"  MAE       : {mae:.4f}\n")
    f.write("=" * 45 + "\n\n")
    f.write(f"Train y mean: {y_train.mean():.2f}\n")
    f.write(f"Test y mean: {y_test.mean():.2f}\n\n")
    f.write("Coefficient Summary (α = 0.05):\n")
    f.write(ols.summary2().tables[1][["Coef.", "Std.Err.", "t", "P>|t|"]].to_string())

# --- Plot: Actual vs Predicted ---
plt.figure(figsize=(12, 4))
plt.plot(y_test.values, label="Actual", linewidth=2)
plt.plot(y_pred, label="Predicted", linewidth=2, linestyle="--")
plt.title("Linear Regression: Actual vs Predicted Consumer Sentiment (Test Set)")
plt.xlabel("Month (test set index)")
plt.ylabel("UMCSENT")
plt.legend()
plt.tight_layout()
plt.savefig("linear_regression_actual_vs_predicted.png", dpi=150, bbox_inches="tight")

# --- Plot: Coefficients ---
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
