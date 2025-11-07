"""
    main.py — Main workflow for the Clustering project

    Purpose:
        - Executes the clustering pipeline step by step.
        - Uses helper functions from utils.py.

    Workflow Steps:
          1. Load dataset
          2. Inspect dataset (info, head, describe, NaN check)
          3. Visual outlier detection (boxplots)
          4. Outlier check per column
          5. Remove outliers in selected columns
          6. Re-evaluate outliers
          7. Check and convert categorical variables (Gender → Gender_code)
          8. Correlation analysis
          9. Scale numeric features
         10. Search for optimal k (Elbow + Silhouette)
         11. KMeans parameter tuning
         12. Final KMeans training
         13. Cluster visualization (2D / 3D)
         14. Analyze clusters with high Spending Score

    Νotes:
        - This script is intended for demonstration and exploratory purposes.
        - Modify `utils.py` to adapt preprocessing or clustering methods for other datasets.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import only the functions you actually use from utils below
from utils import (
    load_data,
    inspect_data,
    plot_all_numeric_boxplots,
    print_outliers_for_all_numeric,
    remove_outliers_iqr,
    get_numeric_dataframe,
    analyze_correlations,
    scale_features,
    evaluate_kmeans_range,
    run_kmeans,
    visualize_clusters,
    analyze_top_cluster,
)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)

# Step 1: Load dataset
df = load_data("datasets/mall_customers.csv")

# Step 2: Inspect dataset
inspect_data(df)

# Step 3: Boxplots for visual outlier detection
plot_all_numeric_boxplots(df, exclude="CustomerID")

# Step 4: Outlier detection using IQR
print_outliers_for_all_numeric(df, exclude="CustomerID")

# Step 5: Remove outliers in the 'Annual Income' column
columns_to_check = ['Annual Income (k$)']
df_cleaned = remove_outliers_iqr(df, columns_to_check)

# Step 6: Re-evaluate outliers after cleaning
plot_all_numeric_boxplots(df_cleaned, exclude="CustomerID")
print_outliers_for_all_numeric(df_cleaned, exclude="CustomerID")

# Step 7: Check unique values & convert Gender to Gender_code
print(df_cleaned["Gender"].unique())
print(df_cleaned["Gender"].value_counts())
df_cleaned["Gender_code"] = df_cleaned["Gender"].map({"Male": 0, "Female": 1})

# Step 8: Correlation analysis
df_corr_input = get_numeric_dataframe(df_cleaned, exclude="CustomerID")
correlations = analyze_correlations(df_corr_input, target="Spending Score (1-100)")

# Step 9: Check distributions before scaling
for column in df_corr_input.select_dtypes(include='number').columns:
    plt.figure(figsize=(6, 4))
    plt.hist(df_corr_input[column].dropna(), bins=30, edgecolor='black')
    plt.title(f'Histogram of {column}')
    plt.xlabel('Τιμή')
    plt.ylabel('Συχνότητα')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Apply scaling to numerical data
scaled_df = scale_features(df_corr_input, method="minmax")

# Step 10: Search for optimal k using Elbow + Silhouette methods
feature_sets = [
    ["Age", "Gender_code", "Annual Income (k$)", "Spending Score (1-100)"],
    ["Age", "Annual Income (k$)", "Spending Score (1-100)"],
    ["Age", "Gender_code", "Spending Score (1-100)"],
    ["Age", "Spending Score (1-100)"]
]
for features in feature_sets:
    print(f"\nΤρέχουμε αξιολόγηση για: {features}")
    X = scaled_df[features]
    evaluate_kmeans_range(X, range(2, 20))

# Step 11: KMeans hyperparameter tuning
params = [
    ('k-means++', 10, 300),
    ('k-means++', 20, 300),
    ('k-means++', 50, 300),
    ('random', 10, 300),
    ('random', 50, 300),
    ('k-means++', 10, 500),
    ('random', 20, 1000),
]
for init_method, n_init, max_iter in params:
    print(f"\nInit: {init_method}, n_init: {n_init}, max_iter: {max_iter}")
    model = run_kmeans(
        scaled_df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]],
        n_clusters=9,
        init_method=init_method,
        n_init=n_init,
        max_iter=max_iter
    )

# Step 12: Final model training
final_model = run_kmeans(
    scaled_df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]],
    n_clusters=9,
)

# Step 13: Cluster visualization
visualize_clusters(final_model, scaled_df, ["Age", "Spending Score (1-100)"], mode="2D")
visualize_clusters(final_model, scaled_df, ["Annual Income (k$)", "Spending Score (1-100)"], mode="2D")
visualize_clusters(final_model, scaled_df, ["Age", "Annual Income (k$)", "Spending Score (1-100)"], mode="3D")

# Step 14: Analysis of clusters with high Spending Score
analyze_cluster = analyze_top_cluster(final_model, df_corr_input, threshold=75, exclude="Gender_code")