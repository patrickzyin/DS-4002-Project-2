import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from statsmodels.tsa.stattools import adfuller

output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "OUTPUT", "lasso_full")
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

X = df.drop(columns=["UMCSENT"])
y = df["UMCSENT"]

adf_raw = adfuller(y)
adf_diff = adfuller(y.diff().dropna())
print(f"ADF Statistic (raw):         {adf_raw[0]:.4f}, p={adf_raw[1]:.4f}")
print(f"ADF Statistic (differenced): {adf_diff[0]:.4f}, p={adf_diff[1]:.4f}")

X_scaled = (X - X.mean()) / X.std()

split = int(len(df) * 0.8)
X_train, X_test = X_scaled.iloc[:split], X_scaled.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

tscv = TimeSeriesSplit(n_splits=5)
lasso = LassoCV(cv=tscv, max_iter=10000)
lasso.fit(X_train, y_train)
y_pred = lasso.predict(X_test)
r2  = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
lasso_s = LassoCV(cv=5, max_iter=10000)
lasso_s.fit(X_train_s, y_train_s)
y_pred_s = lasso_s.predict(X_test_s)
r2_shuffled  = r2_score(y_test_s, y_pred_s)
mae_shuffled = mean_absolute_error(y_test_s, y_pred_s)

coef_df = pd.DataFrame({"Feature": X_train.columns, "Coefficient": lasso.coef_}).sort_values("Coefficient")

print("=" * 45)
print("     Lasso (Full) — Time-Ordered Split")
print("=" * 45)
print(f"  Best lambda (α) : {lasso.alpha_:.4f}")
print(f"  R-squared       : {r2:.4f}  (goal: >= 0.70)")
print(f"  MAE             : {mae:.4f}")
print("=" * 45)
print("=" * 45)
print("     Lasso (Full) — Shuffled Split")
print("=" * 45)
print(f"  Best lambda (α) : {lasso_s.alpha_:.4f}")
print(f"  R-squared       : {r2_shuffled:.4f}")
print(f"  MAE             : {mae_shuffled:.4f}")
print("=" * 45)
print("\nLasso Coefficients:")
print(coef_df.to_string(index=False))

with open(os.path.join(output_dir, "lasso_full_results.txt"), "w", encoding="utf-8") as f:
    f.write("Lasso Regression — Full Dataset (1980–2026)\n")
    f.write("=" * 45 + "\n")
    f.write("   Time-Ordered Split\n")
    f.write("=" * 45 + "\n")
    f.write(f"  Best lambda (α) : {lasso.alpha_:.4f}\n")
    f.write(f"  R-squared       : {r2:.4f}\n")
    f.write(f"  MAE             : {mae:.4f}\n\n")
    f.write(f"ADF (raw): {adf_raw[0]:.4f}, p={adf_raw[1]:.4f}\n")
    f.write(f"ADF (differenced): {adf_diff[0]:.4f}, p={adf_diff[1]:.4f}\n\n")
    f.write("=" * 45 + "\n")
    f.write("   Shuffled Split (diagnostic)\n")
    f.write("=" * 45 + "\n")
    f.write(f"  Best lambda (α) : {lasso_s.alpha_:.4f}\n")
    f.write(f"  R-squared       : {r2_shuffled:.4f}\n")
    f.write(f"  MAE             : {mae_shuffled:.4f}\n\n")
    f.write("Lasso Coefficients:\n")
    f.write(coef_df.to_string(index=False))

plt.figure(figsize=(12, 4))
plt.plot(y_test.values, label="Actual", linewidth=2)
plt.plot(y_pred, label="Predicted", linewidth=2, linestyle="--")
plt.title("Lasso (Full): Actual vs Predicted (Time-Ordered)")
plt.xlabel("Month (test set index)")
plt.ylabel("UMCSENT")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "lasso_full_actual_vs_predicted.png"), dpi=150, bbox_inches="tight")
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(y_test_s.values, label="Actual", linewidth=2)
plt.plot(y_pred_s, label="Predicted", linewidth=2, linestyle="--")
plt.title("Lasso (Full): Actual vs Predicted (Shuffled)")
plt.xlabel("Observation index")
plt.ylabel("UMCSENT")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "lasso_full_actual_vs_predicted_shuffled.png"), dpi=150, bbox_inches="tight")
plt.show()

nonzero = coef_df[coef_df["Coefficient"] != 0]
colors = ["crimson" if c < 0 else "steelblue" for c in nonzero["Coefficient"]]
plt.figure(figsize=(10, 5))
plt.barh(nonzero["Feature"], nonzero["Coefficient"], color=colors)
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Lasso: Non-Zero Coefficients (Full)")
plt.xlabel("Coefficient Value")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "lasso_full_coefficients.png"), dpi=150, bbox_inches="tight")
plt.show()