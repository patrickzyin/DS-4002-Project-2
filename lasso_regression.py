import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score, mean_absolute_error

# --- Load & Clean ---
df = pd.read_csv("Project2-Data.csv")
df = df.drop(columns=["Month"])

# Strip % signs and convert to float
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].str.replace("%", "").astype(float)

# --- Drop highly collinear features ---
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

# --- Lasso with time-series cross-validation ---
# TimeSeriesSplit respects temporal order to avoid data leakage
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
lasso = LassoCV(cv=tscv, max_iter=10000)
lasso.fit(X_train, y_train)

y_pred = lasso.predict(X_test)
r2  = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("=" * 45)
print("          Lasso Regression Results")
print("=" * 45)
print(f"  Best lambda (α) : {lasso.alpha_:.4f}")
print(f"  R-squared       : {r2:.4f}  (goal: >= 0.70)")
print(f"  MAE             : {mae:.4f}")
print("=" * 45)

# --- Coefficients ---
coef_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Coefficient": lasso.coef_
}).sort_values("Coefficient")

print("\nLasso Coefficients (0 = feature eliminated):")
print(coef_df.to_string(index=False))

# --- Save results ---
with open("lasso_regression_results.txt", "w", encoding="utf-8") as f:
    f.write("=" * 45 + "\n")
    f.write("          Lasso Regression Results\n")
    f.write("=" * 45 + "\n")
    f.write(f"  Best lambda (α) : {lasso.alpha_:.4f}\n")
    f.write(f"  R-squared       : {r2:.4f}  (goal: >= 0.70)\n")
    f.write(f"  MAE             : {mae:.4f}\n")
    f.write("=" * 45 + "\n\n")
    f.write("Lasso Coefficients (0 = feature eliminated):\n")
    f.write(coef_df.to_string(index=False))

# --- Plot: Actual vs Predicted ---
plt.figure(figsize=(12, 4))
plt.plot(y_test.values, label="Actual", linewidth=2)
plt.plot(y_pred, label="Predicted", linewidth=2, linestyle="--")
plt.title("Lasso Regression: Actual vs Predicted Consumer Sentiment (Test Set)")
plt.xlabel("Month (test set index)")
plt.ylabel("UMCSENT")
plt.legend()
plt.tight_layout()
plt.savefig("lasso_actual_vs_predicted.png", dpi=150, bbox_inches="tight")

# --- Plot: Coefficients (non-zero only) ---
nonzero = coef_df[coef_df["Coefficient"] != 0]
colors = ["crimson" if c < 0 else "steelblue" for c in nonzero["Coefficient"]]

plt.figure(figsize=(10, 5))
plt.barh(nonzero["Feature"], nonzero["Coefficient"], color=colors)
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Lasso: Non-Zero Standardized Coefficients")
plt.xlabel("Coefficient Value")
plt.tight_layout()
plt.savefig("lasso_coefficients.png", dpi=150, bbox_inches="tight")
