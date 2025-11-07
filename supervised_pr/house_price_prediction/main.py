"""
    Main script for supervised regression on housing prices using linear models.

    Purpose:
        - Runs an end-to-end supervised workflow: data loading/inspection,
          cleaning and feature engineering, train/test splitting, model training,
          evaluation, cross-validation, and visualization.
        - Compares a simple linear regression (two features) with a composite model
          that includes engineered features and location dummies.

    Workflow Overview:
        1. Load dataset from Excel (.xls) and perform basic inspection.
        2. Clean column names and construct the target variable `price`.
        3. Explore numeric features: boxplots, outlier tables, distributions, skew/kurtosis.
        4. Compute correlations against `price`.
        5. Model 1 (Simple): select two features, split (90/10), fit LinearRegression,
           report coefficients/intercept, visualize actual vs predicted, and compute RMSE/MAE/R².
        6. Create engineered features (f1–f5) and one-hot encode `location` dummies.
        7. Model 2 (Composite): split (90/10), fit LinearRegression, print coefficients,
           visualize actual vs predicted, and compute RMSE/MAE/R².
        8. Perform 5-fold cross-validation for both models and compare mean RMSE/MAE.
        9. Visual comparisons: bar charts for mean RMSEs and a three-bar plot
           for a custom 5-house test block.

    Notes:
        - Reusable utilities (loading, inspection, plots, CV metrics) live in `utils.py`.
        - Adjust feature engineering and scaling strategies in this script as needed.
        - Pandas display options are optional and can be tuned for wider tables.
"""
# --- Basic libs ---
import pandas as pd
import matplotlib.pyplot as plt

# Pandas display options (προαιρετικά)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)

# --- ML ---
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

# --- Project utilities ---
from utils import (
    load_data,
    inspect_data,
    plot_all_numeric_boxplots,
    print_outliers_for_all_numeric,
    get_numeric_dataframe,
    analyze_correlations,
    plot_distributions,
    show_skew_kurtosis,
    clean_and_convert_column,
    clean_column_names,
    cross_validated_rmse,
)

# ===========================
# Start of workflow
# ===========================

# Load dataset from Excel (.xls) with a given sheet
df, target = load_data(
    source='file',
    filepath="datasets/housedata.xls",
    dataset_func=None,
    sheet_name="Sheet1"
)

# Basic inspection
inspect_data(df)

# Clean column names
clean_column_names(df)

# Clean and convert selling price column; create 'price'
df['price'] = clean_and_convert_column(df["selling_price_in_1000_dollars"])

# Correlations against target 'price' over numeric-only DataFrame
df_numeric = get_numeric_dataframe(df, exclude=None)
analyze_correlations(df_numeric, target="price")

# ===========================
# Simple linear model: X_1 with two features, y_1 as price
# ===========================

# Select features and target for the first model
X_1 = df[["house_area_in_1000_square_feet", "bedrooms"]]
y_1 = df["price"]

# Outlier checks (visual + tabular)
plot_all_numeric_boxplots(X_1, exclude=None)
print_outliers_for_all_numeric(X_1, exclude=None)

# Distribution plots and skew/kurtosis stats for inputs
plot_distributions(X_1)
show_skew_kurtosis(X_1)

# Train/test split (approx. 90/10 as per the notebook description)
X_1_train, X_1_test, y_1_train, y_1_test = train_test_split(
    X_1, y_1, test_size=0.1, random_state=42
)

print("Training set:", X_1_train.shape)
print("Test set:", X_1_test.shape)

# Fit simple linear regression
model_1 = LinearRegression()
model_1.fit(X_1_train, y_1_train)

# Coefficients and intercept
print("β1 (συντελεστής x1 - house area):", model_1.coef_[0])
print("β2 (συντελεστής x2 - bedrooms):", model_1.coef_[1])
print("c (σταθερά):", model_1.intercept_)

# ===========================
# Actual vs predicted on all data (scatter)
# ===========================
y_pred_all_1 = model_1.predict(X_1)

plt.figure(figsize=(8, 6))
plt.scatter(y_1, y_pred_all_1, alpha=0.6)
plt.xlabel("Πραγματική Τιμή (y)")
plt.ylabel("Προβλεπόμενη Τιμή (ŷ)")
plt.title("Πραγματική vs Προβλεπόμενη Τιμή Πώλησης για Όλα τα Σπίτια")
plt.grid(True)
plt.plot([y_1.min(), y_1.max()], [y_1.min(), y_1.max()], 'r--')
plt.show()


# ===========================
# Custom 5-house test block (kept as-is)
# ===========================
test_houses = pd.DataFrame({
    'house_area_in_1000_square_feet': [846, 1324, 1150, 3037, 3984],
    'bedrooms': [1, 2, 3, 4, 5],
    'actual_price': [115000, 234500, 198000, 528000, 572500]
})

X_custom = test_houses[['house_area_in_1000_square_feet', 'bedrooms']]
test_houses['predicted_price'] = model_1.predict(X_custom)

plt.figure(figsize=(8, 6))
plt.scatter(test_houses["actual_price"], test_houses["predicted_price"], color='orange', s=80)
plt.plot(
    [test_houses["actual_price"].min(), test_houses["actual_price"].max()],
    [test_houses["actual_price"].min(), test_houses["actual_price"].max()],
    'r--'
)
plt.xlabel("Πραγματική Τιμή (y)")
plt.ylabel("Προβλεπόμενη Τιμή (ŷ)")
plt.title("Πρόβλεψη Τιμής για τα 5 Σπίτια")
plt.grid(True)
plt.show()

rmse_custom = root_mean_squared_error(test_houses["actual_price"], test_houses["predicted_price"])
mae_custom = mean_absolute_error(test_houses["actual_price"], test_houses["predicted_price"])

print(f"RMSE στα 5 σπίτια: {rmse_custom:.2f}")
print(f"MAE στα 5 σπίτια: {mae_custom:.2f}\n")

print("Πραγματική vs Προβλεπόμενη Τιμή:")
for i in range(len(test_houses)):
    actual = test_houses.loc[i, "actual_price"]
    predicted = test_houses.loc[i, "predicted_price"]
    print(f"Σπίτι {i+1}: Πραγματική = {actual:,.0f} | Προβλεπόμενη = {predicted:,.0f}")

# ===========================
# Train/test metrics for Model 1
# ===========================
y_train_pred_1 = model_1.predict(X_1_train)
y_test_pred_1 = model_1.predict(X_1_test)

rmse_train_1 = root_mean_squared_error(y_1_train, y_train_pred_1)
rmse_test_1 = root_mean_squared_error(y_1_test, y_test_pred_1)

mae_train_1 = mean_absolute_error(y_1_train, y_train_pred_1)
mae_test_1 = mean_absolute_error(y_1_test, y_test_pred_1)

r2_train_2 = r2_score(y_1_train, y_train_pred_1)
r2_test_2 = r2_score(y_1_test, y_test_pred_1)

print(f" RMSE (Train): {rmse_train_1:.2f} ")
print(f" RMSE (Test): {rmse_test_1:.2f} ")
print("\n--------------------")
print(f" MAE (Train): {mae_train_1:.2f} ")
print(f" MAE (Test): {mae_test_1:.2f} ")
print("\n--------------------")
print(f" R2 (Train): {r2_train_2:.2f} ")
print(f" R2 (Test): {r2_test_2:.2f} ")

# ===========================
# Feature engineering for the second (composite) model
# ===========================
location_dummies = pd.get_dummies(df["location"], prefix="loc", drop_first=True)

df["f1"] = 1
df["f2"] = df["house_area_in_1000_square_feet"]
df["f3"] = df["house_area_in_1000_square_feet"].apply(lambda x: max(x - 1500, 0))
df["f4"] = df["bedrooms"]
df["f5"] = df["1_if_condo_0_otherwise"]

features_to_scale = df[["f2", "f3", "f4"]]
X_2 = pd.concat([df[["f1"]], features_to_scale, df[["f5"]], location_dummies], axis=1)
y_2 = df["price"]

print(X_2.columns.tolist())

# Correlations for engineered features vs price (visual)
X_2_corr = pd.concat([df[["f1", "f2", "f3", "f4", "f5"]], location_dummies, df["price"]], axis=1)
analyze_correlations(X_2_corr, target="price")

# ===========================
# Train/test split and training for Model 2
# ===========================
X_2_train, X_2_test, y_2_train, y_2_test = train_test_split(X_2, y_2, test_size=0.1, random_state=42)

model_2 = LinearRegression()
model_2.fit(X_2_train, y_2_train)

print("Συντελεστές μοντέλου (β1 έως β8):\n")
coef_labels = [
    "β1 (f1 - σταθερά βάσης)",
    "β2 (f2 - house area)",
    "β3 (f3 - max(area - 1500, 0))",
    "β4 (f4 - bedrooms)",
    "β5 (f5 - is_condo)",
    "β6 (f6 - location=2)",
    "β7 (f7 - location=3)",
    "β8 (f8 - location=4)"
]
for i, coef in enumerate(model_2.coef_):
    print(f"{coef_labels[i]}: {coef:.2f}")
print(f"\nc (σταθερός όρος): {model_2.intercept_:.2f}")

# Actual vs predicted plot for Model 2
y_pred_2 = model_2.predict(X_2)
plt.figure(figsize=(8, 6))
plt.scatter(y_2, y_pred_2, alpha=0.5, label="Predicted vs Actual")
plt.plot([y_2.min(), y_2.max()], [y_2.min(), y_2.max()], 'r--', label="Ideal (y = ŷ)")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(" New Regression Model: Actual vs Predicted")
plt.legend()
plt.grid(True)
plt.show()

# Metrics for Model 2
y_train_pred_2 = model_2.predict(X_2_train)
y_test_pred_2 = model_2.predict(X_2_test)

rmse_train_2 = root_mean_squared_error(y_2_train, y_train_pred_2)
rmse_test_2 = root_mean_squared_error(y_2_test, y_test_pred_2)

mae_train_2 = mean_absolute_error(y_2_train, y_train_pred_2)
mae_test_2 = mean_absolute_error(y_2_test, y_test_pred_2)

r2_train_2 = r2_score(y_2_train, y_train_pred_2)
r2_test_2 = r2_score(y_2_test, y_test_pred_2)

print(f" RMSE (Train): {rmse_train_2:.2f} ")
print(f" RMSE (Test): {rmse_test_2:.2f} ")
print("\n--------------------")
print(f" MAE (Train): {mae_train_2:.2f} ")
print(f" MAE (Test): {mae_test_2:.2f} ")
print("\n--------------------")
print(f" R2 (Train): {r2_train_2:.2f} ")
print(f" R2 (Test): {r2_test_2:.2f} ")

# ===========================
# Cross-validation (5-fold) for both models
# ===========================
rmse_scores_1, mean_rmse_1, mae_scores_1, mean_mae_1 = cross_validated_rmse(model_1, X_1, y_1, return_mean=True)
for rmse in rmse_scores_1:
    print("\nRMSE ανά fold για το πρώτο μοντέλο:", rmse)
print("\n-----------------\nΜέση RMSE για το πρώτο μοντέλο:", mean_rmse_1)
print("\n-----------------")
for mae in mae_scores_1:
    print("\nMAE ανά fold για το πρώτο μοντέλο:", mae)
print("\n-----------------\nΜέση MAE για το πρώτο μοντέλο:", mean_mae_1)
print("\n-----------------")

rmse_scores_2, mean_rmse_2, mae_scores_2, mean_mae_2 = cross_validated_rmse(model_2, X_2, y_2, return_mean=True)
for rmse in rmse_scores_2:
    print("\nRMSE ανά fold για το δεύτερο μοντέλο:", rmse)
print("\n-----------------\nΜέση RMSE για το δεύτερο μοντέλο:", mean_rmse_2)
print("\n-----------------")
for mae in mae_scores_2:
    print("\nMAE ανά fold για το δεύτερο μοντέλο:", mae)
print("\n-----------------\nΜέση MAE για το δεύτερο μοντέλο:", mean_mae_2)
print("\n-----------------")

# Bar plot comparison of mean RMSEs
rmse_model_1 = rmse_scores_1
rmse_model_2 = rmse_scores_2

models = ['Απλό Μοντέλο', 'Σύνθετο Μοντέλο']
mean_rmses = [mean_rmse_1, mean_rmse_2]

plt.figure(figsize=(8, 6))
bars = plt.bar(models, mean_rmses, color=['skyblue', 'lightgreen'])
plt.ylabel('Μέση RMSE')
plt.title('Σύγκριση Μέσης RMSE ανά Μοντέλο (5-fold Cross-Validation)')
plt.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 1000, f'{height:.0f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Three-bar comparison for the 5 custom houses (kept as in notebook)
houses = ['Σπίτι 1', 'Σπίτι 2', 'Σπίτι 3', 'Σπίτι 4', 'Σπίτι 5']
true = [115000, 234500, 198000, 528000, 572500]
pred_no = [156013, 210031, 167774, 429847, 553118]
pred_minmax = [110681, 174639, 123115, 436632, 583659]

x = range(len(houses))

plt.bar([i - 0.25 for i in x], true, width=0.25, label='Πραγματική Τιμή')
plt.bar(x, pred_no, width=0.25, label='No Scaling')
plt.bar([i + 0.25 for i in x], pred_minmax, width=0.25, label='MinMax Scaling')

plt.xticks(list(x), houses)
plt.ylabel('Τιμή (€)')
plt.title('Πραγματική vs Προβλεπόμενη Τιμή ανά Σπίτι')
plt.legend()
plt.show()