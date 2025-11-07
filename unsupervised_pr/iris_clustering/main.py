"""
    Main script for clustering analysis using multiple algorithms on the Iris dataset.

    Purpose:
        - Executes the full end-to-end workflow: data loading, exploration, outlier handling,
          scaling, clustering (K-Means, Agglomerative, HDBSCAN), and visual evaluation.
        - Demonstrates the impact of data cleaning and feature scaling on clustering performance.
        - Compares clustering models quantitatively (NMI, AMI, ARI) and visually across methods.
        - Keeps all reusable utility functions in `utils.py` for modularity and clarity.

    Workflow Overview:
        1. Load the Iris dataset.
        2. Inspect and visualize raw data (distributions, boxplots, correlations).
        3. Detect and remove outliers using IQR.
        4. Analyze correlations and skewness/kurtosis before and after cleaning.
        5. Apply feature scaling (StandardScaler).
        6. Run and evaluate K-Means across k values.
        7. Train K-Means and visualize clusters.
        8. Perform Agglomerative clustering and plot dendrograms.
        9. Compare Agglomerative scenarios (k=2, k=3, distance threshold).
        10. Run HDBSCAN and visualize cluster structures.
        11. Compare all clustering methods side by side.
        12. Evaluate model quality with ground truth labels.
        13. Repeat key experiments with outliers included.

    Notes:
        - This script is intended for demonstration and exploratory purposes.
        - Modify `utils.py` to adapt preprocessing or clustering methods for other datasets.
"""

# ------------------- Εισαγωγές μόνο για τη ροή -------------------
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from utils import (
    load_data, inspect_data,
    plot_all_numeric_boxplots, print_outliers_for_all_numeric, remove_outliers_iqr,
    analyze_correlations, plot_distributions, show_skew_kurtosis,
    scale_features, evaluate_kmeans_range, run_kmeans,
    evaluate_agglomerative_range, plot_dendrogram_only, run_agglomerative_clustering,
    run_hdbscan_clustering, visualize_clusters, plot_cluster_results,
    evaluate_clustering_models
)

# --- pandas display settings ---
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)

# ------------------- Step 1: Load Iris dataset -------------------
df, target = load_data('sklearn', dataset_func=load_iris)

# ------------------- Step 2: Data inspection -------------------
inspect_data(df)

# ------------------- Step 3: Outliers (boxplots + IQR) -------------------
plot_all_numeric_boxplots(df)
print_outliers_for_all_numeric(df)

# ------------------- Create a cleaned version without outliers -------------------
df_cleaned = remove_outliers_iqr(df, df.columns)
df_cleaned = df_cleaned.reset_index(drop=True)
target_cleaned = target[df_cleaned.index]

# ------------------- Step 4: Correlations (with & without outliers) -------------------
analyze_correlations(df)           # με outliers
analyze_correlations(df_cleaned)   # χωρίς outliers

# ------------------- Step 5: Distributions + Skewness/Kurtosis -------------------
plot_distributions(df_cleaned)
show_skew_kurtosis(df_cleaned)

# (Optional: also on the original DataFrame)
plot_distributions(df)
show_skew_kurtosis(df)

# ------------------- Step 6: Scaling (StandardScaler) -------------------
df_scaled = scale_features(df, method="standard")
df_scaled_cleaned = scale_features(df_cleaned, method="standard")

# ------------------- Step 7: KMeans — finding k -------------------
evaluate_kmeans_range(df_scaled_cleaned, k_range=range(2, 11))

# ------------------- Step 8: KMeans — training with k=2 -------------------
kmeansmodel = run_kmeans(df_scaled_cleaned, n_clusters=2)

# ------------------- Step 9: KMeans visualization (all 2D feature pairs) -------------------
feature_pairs = [
    ["sepal length (cm)", "sepal width (cm)"],
    ["sepal length (cm)", "petal length (cm)"],
    ["sepal length (cm)", "petal width (cm)"],
    ["sepal width (cm)", "petal length (cm)"],
    ["sepal width (cm)", "petal width (cm)"],
    ["petal length (cm)", "petal width (cm)"]
]
for pair in feature_pairs:
    visualize_clusters(kmeansmodel, df_scaled_cleaned, pair, mode="2D")

# ------------------- Step 10: Agglomerative — k range + dendrogram -------------------
evaluate_agglomerative_range(df_scaled_cleaned, k_range=range(2, 11), linkage='ward', metric='euclidean', show_scores=True)
plot_dendrogram_only(df_scaled_cleaned, linkage_method='ward', metric='euclidean', figsize=(12, 6), title="Dendrogram", truncate_lastp=12)

# ------------------- Step 11: Agglomerative — three scenarios -------------------
agg_model_k2 = run_agglomerative_clustering(df_scaled_cleaned, n_clusters=2)
agg_model_k3 = run_agglomerative_clustering(df_scaled_cleaned, n_clusters=3)
agg_model_d10 = run_agglomerative_clustering(df_scaled_cleaned, distance_threshold=10, n_clusters=None)

# ------------------- Step 12: Visual comparison of Agglomerative (2D, all pairs) -------------------
agg_models = [agg_model_k2, agg_model_k3, agg_model_d10]
titles = ["k = 2", "k = 3", "distance_threshold = 10"]
plot_cluster_results(df_scaled_cleaned, agg_models, titles, feature_pairs)

# ------------------- Step 13: HDBSCAN — run & 2D visualization -------------------
hdbscan_model = run_hdbscan_clustering(df_scaled_cleaned)
for pair in feature_pairs:
    visualize_clusters(hdbscan_model, df_scaled_cleaned, pair, mode="2D")

# ------------------- Step 14: Visual comparison — KMeans vs Agglomerative vs HDBSCAN -------------------
models = [kmeansmodel, agg_model_k2, hdbscan_model]
titles = ["KMeans (k = 2)", "Agglomerative (k = 2)", "HDBSCAN"]
plot_cluster_results(df_scaled_cleaned, models, titles, feature_pairs)

# ------------------- Step 15: Metrics — NMI/AMI/ARI (without outliers) -------------------
models_eval = [kmeansmodel, agg_model_k2, agg_model_k3, hdbscan_model]
titles_eval = ["KMeans (k = 2)", "Agglomerative (k = 2)", "Agglomerative (k = 3)", "HDBSCAN"]
true_labels_list = [target_cleaned, target_cleaned, target_cleaned, target_cleaned]
evaluate_clustering_models(true_labels_list, models_eval, titles_eval)

# ------------------- Step 16: Experiments with outliers (same routines) -------------------
kmeansmodel_no_cln = run_kmeans(df_scaled, n_clusters=2)
agg_model_k2_no_cln = run_agglomerative_clustering(df_scaled, n_clusters=2)
agg_model_k3_no_cln = run_agglomerative_clustering(df_scaled, n_clusters=3)
agg_model_d10_no_cln = run_agglomerative_clustering(df_scaled, distance_threshold=10, n_clusters=None)
hdbscan_model_no_cln = run_hdbscan_clustering(df_scaled)

models_no_cln = [kmeansmodel_no_cln, agg_model_k2_no_cln, agg_model_k3_no_cln, hdbscan_model_no_cln]
titles_no_cln = ["KMeans (k = 2)", "Agglomerative (k = 2)", "Agglomerative (k = 3)", "HDBSCAN"]
true_labels_list_no_cln = [target, target, target, target]
evaluate_clustering_models(true_labels_list_no_cln, models_no_cln, titles_no_cln)