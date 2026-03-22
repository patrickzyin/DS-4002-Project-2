# DS 4002 — Project 2: CPI Components and Consumer Sentiment

## Section 0: Project Overview

This project investigates which specific components of the Consumer Price Index (CPI) act as the primary drivers of negative consumer sentiment in the United States. Using monthly CPI data from the U.S. Bureau of Labor Statistics and the University of Michigan Consumer Sentiment Index (UMCSENT) spanning 1980–2026, we apply Linear Regression, Lasso Regression, and ARIMAX models to identify the most influential price categories and evaluate their predictive power across stable and crisis-affected economic periods.

**Hypothesis:**
Shelter inflation will be the primary driver of decreased consumer sentiment.

**Research Question:**
Which specific component of the Consumer Price Index (CPI) acts as the primary driving factor for increased negative consumer sentiment toward the broader economy?


## Section 1: Software and Platform

**Language:** Python 3.11+ 

**Required packages** — install all via pip or conda before running any scripts:

```
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
```

| Package | Version Used | Purpose |
|---|---|---|
| pandas | 2.2.3 | Data loading and manipulation |
| numpy | 2.1.3 | Numerical operations |
| matplotlib | 3.10.0 | Plotting |
| seaborn | 0.13.2 | Correlation matrix heatmap |
| scikit-learn | 1.4+ | Linear regression, Lasso, train/test split, metrics |
| statsmodels | 0.14+ | OLS t-tests, ARIMA/ARIMAX, ADF stationarity tests |

## Section 2: Map of Documentation

DS-4002-Project-2/
* README.md — Project overview, reproduction instructions, and results summary

DATA/
* Project2-Data.csv — Combined dataset containing monthly CPI component percentage changes and UMCSENT scores from January 1980 to January 2026. Columns: Year, UMCSENT, Overall CPI, Food, Shelter, Apparel, Gasoline (all types), Electricity, Natural Gas, Commodities Less Food and Energy Commodities, New Vehicles, Medical Care Commodities, Services Less Energy Services, Medical Care services, Core CPI
* Project2-RawData.csv - Unedited original combined dataset from January 2001 to January 2026. Columns: Month, UMCSENT, Food, Food at Home, Food Away from Home, Energy, Gasoline (all types), Electricity, Natural Gas (piped), All items less food and energy, Commodities less food and energy commodoties, Apparel, New Vehicles, Medical Care Commodoties, Shelter less energy services, Shelter, Medical Care Services, Education and Communication

SCRIPTS/
* correlation.ipynb — Generates a heatmap of correlations between all CPI features and prints feature pairs with correlation above 0.85, used to identify which columns to drop before modeling
* linear_regression_full.py — Runs OLS linear regression on the full 1980–2026 dataset with both time-ordered and shuffled splits, outputs coefficients, R-squared, MAE, and plots
* linear_regression_precovid.py — Same as above but filtered to data before April 2020 to isolate pre-COVID relationships
* lasso_full.py — Runs Lasso regression with time-series cross-validation on the full dataset, automatically selects the most important CPI features
* lasso_precovid.py — Same as above but filtered to pre-COVID data
* arimax_full.py — Fits an ARIMAX(1,1,1) time series model on the full dataset using Food as the exogenous feature (selected by Lasso), produces a cold forecast with 95% confidence intervals
* arimax_precovid.py — Same as above on pre-COVID data using the 8 features selected by Lasso: Food, Medical Care services, Gasoline, Electricity, Commodities Less Food and Energy, Natural Gas, Medical Care Commodities, Shelter

OUTPUT/
* correlation_matrix/correlation_matrix.png — Heatmap of feature correlations
* linear_regression_full/linear_regression_full_results.txt — R-squared, MAE, ADF stationarity results, and full coefficient table for the full dataset linear regression
* linear_regression_full/linear_full_actual_vs_predicted.png — Actual vs predicted plot (time-ordered)
* linear_regression_full/linear_full_actual_vs_predicted_shuffled.png — Actual vs predicted plot (shuffled diagnostic)
* linear_regression_full/linear_full_coefficients.png — Standardized coefficient bar chart
* linear_regression_precovid/linear_regression_precovid_results.txt — Same as above for pre-COVID dataset
* linear_regression_precovid/linear_precovid_actual_vs_predicted.png — Actual vs predicted plot (time-ordered)
* linear_regression_precovid/linear_precovid_actual_vs_predicted_shuffled.png — Actual vs predicted plot (shuffled diagnostic)
* linear_regression_precovid/linear_precovid_coefficients.png — Standardized coefficient bar chart
* lasso_full/lasso_full_results.txt — Lambda, R-squared, MAE, and Lasso coefficients for full dataset
* lasso_full/lasso_full_actual_vs_predicted.png — Actual vs predicted plot (time-ordered)
* lasso_full/lasso_full_actual_vs_predicted_shuffled.png — Actual vs predicted plot (shuffled diagnostic)
* lasso_full/lasso_full_coefficients.png — Non-zero Lasso coefficient bar chart
* lasso_precovid/lasso_precovid_results.txt — Same as above for pre-COVID dataset
* lasso_precovid/lasso_precovid_actual_vs_predicted.png — Actual vs predicted plot (time-ordered)
* lasso_precovid/lasso_precovid_actual_vs_predicted_shuffled.png — Actual vs predicted plot (shuffled diagnostic)
* lasso_precovid/lasso_precovid_coefficients.png — Non-zero Lasso coefficient bar chart
* arimax_full/arimax_full_results.txt — R-squared, MAE, ADF results, and SARIMAX model summary for full dataset
* arimax_full/arimax_full_cold_forecast.png — Full series plot with cold forecast and 95% confidence interval
* arimax_full/arimax_full_actual_vs_predicted.png — Test set forecast vs actual with confidence interval
* arimax_precovid/arimax_precovid_results.txt — Same as above for pre-COVID dataset
* arimax_precovid/arimax_precovid_cold_forecast.png — Full series plot with cold forecast and 95% confidence interval
* arimax_precovid/arimax_precovid_actual_vs_predicted.png — Test set forecast vs actual with confidence interval

## Section 3: Reproducing the Results

Follow these steps in order. All scripts are run from inside the `SCRIPTS/` folder.

### Step 1 — Set up your environment

1. Clone or download this repository and navigate into the project folder:
   ```
   cd DS-4002-Project-2
   ```
2. Install required Python packages:
    ```
    pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
    ```

### Step 2 — Verify the data file

Make sure `Project2-Data.csv` is located in the `DATA/` folder. The file should contain monthly observations from January 1980 through January 2026 with the following columns: Year, UMCSENT, Overall CPI, Food, Shelter, Apparel, Gasoline (all types), Electricity, Natural Gas, Commodities Less Food and Energy Commodities, New Vehicles, Medical Care Commodities, Services Less Energy Services, Medical Care services, and Core CPI.

### Step 3 — Run the correlation matrix

Navigate into the SCRIPTS folder and run:
```
cd SCRIPTS
python correlation_matrix.py
```
This generates a heatmap of correlations between all CPI features and prints any pairs with correlation above 0.85. The output image saves to `OUTPUT/correlation_matrix/`. This step identifies which features to drop for multicollinearity before running the models — the following columns are dropped in all subsequent scripts: Overall CPI, Core CPI, Energy, Services Less Energy Services, Food at home, Food away from home.

### Step 4 — Run the linear regression models

Run the full dataset version first, then the pre-COVID version:
```
python linear_regression_full.py
python linear_regression_precovid.py
```
Each script prints results to the terminal and saves a results text file and three plots to its respective subfolder in `OUTPUT/`. The full dataset uses all data from 1980–2026. The pre-COVID version filters to data before April 2020. Both run an 80/20 time-ordered split and a shuffled diagnostic split. Runtime is under 30 seconds each.

### Step 5 — Run the Lasso regression models

```
python lasso_full.py
python lasso_precovid.py
```
These work identically to the linear regression scripts but use Lasso regularization with time-series cross-validation to automatically select the most important features. Outputs save to `OUTPUT/lasso_full/` and `OUTPUT/lasso_precovid/`. Runtime is under 60 seconds each.

### Step 6 — Run the ARIMAX models

```
python arimax_full.py
python arimax_precovid.py
```
These scripts fit an ARIMAX(1,0,1) model — a time series model that combines sentiment's own historical momentum with CPI components as external inputs. The model is trained on the first 80% of the data and makes a cold forecast on the remaining 20% with no updates, producing a forecast with 95% confidence intervals. The pre-COVID version uses the 8 features selected by Lasso (Food, Medical Care services, Gasoline, Electricity, Commodities Less Food and Energy, Natural Gas, Medical Care Commodities, Shelter). The full dataset version uses only Food, which was the sole feature retained by Lasso on the full dataset. Outputs save to `OUTPUT/arimax_full/` and `OUTPUT/arimax_precovid/`. Runtime is under 2 minutes each.
