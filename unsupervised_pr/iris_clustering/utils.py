"""
    utils.py — Helper functions for the clustering and data analysis project

    Purpose
        - Centralizes ALL reusable functions related to data loading, preprocessing,
          exploratory analysis, scaling, outlier detection, clustering, and visualization.
        - Keeps the main execution logic flow inside main.py for a clean, modular structure.

    Contents (indicative)
        - load_data, inspect_data
        - plot_* (distributions, boxplots, scatter plots, dendrograms)
        - detect/remove outliers (IQR-based)
        - analyze_correlations, show_skew_kurtosis
        - scale_features (StandardScaler, MinMaxScaler, etc.)
        - run_kmeans, evaluate_kmeans_range, visualize_clusters
        - run_agglomerative_clustering, plot_dendrogram_only
        - run_hdbscan_clustering, compare_clustering_results
        - evaluate_clustering_models (NMI, AMI, ARI)

    Usage
          from utils import (
              load_data, inspect_data, analyze_correlations,
              scale_features, run_kmeans, visualize_clusters,
              evaluate_clustering_models
          )

    Dependencies
        - pandas, numpy, matplotlib, seaborn, scikit-learn, hdbscan, scipy
        - Optionally: IPython.display.display (for DataFrame visualization)

    Notes
        - This file defines functions only — no execution flow occurs here.
        - All dataset paths, model parameters, and workflow logic should be defined in main.py.
        - Each function is designed to be modular and reusable across datasets.

    Maintenance
        - When adding new preprocessing, clustering, or visualization routines,
          define them here and import explicitly from main.py.
        - Keep functions self-contained and dataset-agnostic for easier reuse.
"""

# ------------------- Imports for utilities -------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# For converting from Jupyter notebook to .py script
from IPython.display import display

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer
from sklearn.cluster import KMeans, AgglomerativeClustering, HDBSCAN
from sklearn.metrics import silhouette_score
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score, adjusted_rand_score

from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage as scipy_linkage

def load_data(source='file', filepath=None, dataset_func=None):
    """
    Loads data either from a CSV file or from an sklearn.datasets function (e.g., load_iris).

    Parameters:
        source (str): 'file' or 'sklearn'
        filepath (str): Path to the CSV file (if source='file')
        dataset_func (function): sklearn function that loads the dataset (if source='sklearn')

    Returns:
        df (DataFrame): The data as a pandas DataFrame
        target (ndarray or None): The target labels, if available; otherwise None
    """
    if source == 'file':
        if not filepath:
            print("\nΠαρακαλώ δώσε filepath για CSV.")
            return None, None
        try:
            df = pd.read_csv(filepath)
            print("\n Dataset φορτώθηκε από αρχείο:", df.shape)
            return df, None
        except Exception as e:
            print("\nΣφάλμα κατά το διάβασμα του αρχείου:", e)
            return None, None

    elif source == 'sklearn':
        if not dataset_func:
            print("\nΠαρακαλώ δώσε συνάρτηση π.χ. load_iris για φόρτωση sklearn dataset.")
            return None, None
        try:
            dataset = dataset_func()
            df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
            target = dataset.target
            print(f"\nDataset φορτώθηκε από sklearn ({dataset_func.__name__}):", df.shape)
            return df, target
        except Exception as e:
            print(f"\nΣφάλμα κατά το φόρτωμα του Dataset:", e)
            return None, None

    else:
        print("\nΜη έγκυρη επιλογή source. Δοκίμασε: 'file' ή 'sklearn'.")
        return None, None

def inspect_data(df):
    """
    Displays basic information about the dataset:
    - Shape (rows, columns)
    - Data types & NaN values
    - First few rows
    - Descriptive statistics
    """
    print(f"\nΣχήμα DataFrame: {df.shape}")
    print("\nΠληροφορίες DataFrame:")
    df.info()
    print("\nΠρώτες 5 γραμμές:")
    print(df.head())
    print("\nΠεριγραφικά στατιστικά:")
    print(df.describe())
    print("\nΈλεγχος για Nan:")
    print(df.isna().sum())


def plot_single_boxplot(column_data, column_name):
    """
    Displays a horizontal boxplot for a given column and calculates IQR-based thresholds.

    Parameters:
        column_data (Series): The column to visualize (e.g., df["Age"])
        column_name (str): The name to display in the title
    """
    Q1 = column_data.quantile(0.25)
    Q3 = column_data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - (1.5 * IQR)
    upper_bound = Q3 + (1.5 * IQR)

    fig = plt.figure(figsize=(8, 2))
    sns.boxplot(x=column_data)
    plt.title(f"Boxplot - {column_name}")
    plt.xlabel(column_name)
    plt.grid(True)
    # plt.show

    display(fig)
    plt.close(fig)  # to prevent the figure from being displayed again

    print(f"\nΣτατιστικά για {column_name}")
    print(f"\nQ1 (25%): {Q1:.2f}")
    print(f"\nQ3 (75%): {Q3:.2f}")
    print(f"\nIQR: {IQR:.2f}")
    print(f"\nΠιθανοί outliers κάτω από: {lower_bound:.2f}")
    print(f"\nΠιθανοί outliers πάνω από: {upper_bound:.2f}")

def get_numeric_features(df, exclude=None):
    """
    Returns the numeric columns of the DataFrame.

    Parameters:
        df (DataFrame): The dataset containing the columns.
        exclude (str): (Optional) Name of a column to exclude from the numeric set.

    Returns:
        Index: Numeric columns excluding the specified column (if provided).
    """
    # Find columns with numeric data types (int, float, etc.)
    numeric_columns = df.select_dtypes(include=["number"]).columns

    # If an exclusion column was provided and exists, remove it
    if exclude in numeric_columns:
        numeric_columns = numeric_columns.drop(exclude)

    # Return numeric columns
    return numeric_columns


def plot_all_numeric_boxplots(df, exclude=None):
    """
    Creates boxplots for each numeric column in the DataFrame,
    excluding the one specified in the 'exclude' parameter (e.g., 'CustomerID').

    Parameters:
        df (DataFrame): The dataset.
        exclude (str): (Optional) Name of the column to exclude.
    """
    # Isolate numeric columns and remove the excluded one (e.g., customer ID)
    numeric_columns = get_numeric_features(df, exclude)

    # Display a boxplot and quantiles for each numeric column
    for col in numeric_columns:
        plot_single_boxplot(df[col], col)


def find_outliers_iqr(df, column):
    """
    Returns the rows of the DataFrame that are outliers
    in the specified column based on the 1.5 * IQR rule.

    Parameters:
        df (DataFrame): The dataset.
        column (str): The name of the column to check.

    Returns:
        DataFrame: The rows identified as outliers.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - (1.5 * IQR)
    upper_bound = Q3 + (1.5 * IQR)

    # Outlier filtering
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]

    return outliers


def print_outliers_for_all_numeric(df, exclude=None):
    """
    Checks for outliers in all numeric columns (excluding those specified)
    and displays the outlier rows for each column separately.

    Parameters:
        df (DataFrame): The DataFrame containing the data.
        exclude (str): (Optional) Column to exclude, e.g., 'CustomerID'.
    """
    # Identify the columns that are numeric types (int, float, etc.)
    numeric_columns = df.select_dtypes(include=["number"]).columns

    # If an exclusion column is specified and exists, remove it from the list
    if exclude in numeric_columns:
        numeric_columns = numeric_columns.drop(exclude)

    for col in numeric_columns:
        print(f"\nΈλεγχος outliers στην στήλη:{col}")
        outliers = find_outliers_iqr(df, col)
        if outliers.empty:
            print("\nΔεν βρέθηκαν outliers")
        else:
            print(f"\nΒρέθηκαν {len(outliers)} outliers:")
            print(outliers)

def remove_outliers_iqr(df, columns):
    """
        Removes all rows that are outliers in any of the given columns,
        based on the IQR rule.

        Parameters:
            df (DataFrame): The original dataset.
            columns (list): List of column names to check for outliers.

        Returns:
            DataFrame: A new DataFrame without the outliers.
        """
    # Use a set to avoid duplicate indices
    outlier_index = set()
    for col in columns:
        outliers = find_outliers_iqr(df, col)
        outlier_index.update(outliers.index)
    df_cleaned = df.drop(index=outlier_index)
    print(f"Αφαιρέθηκαν {len(df) - len(df_cleaned)} γραμμές με outliers.")
    return df_cleaned


def get_numeric_dataframe(df, exclude=None):
    """
    Returns a new DataFrame containing only numeric columns.
    If an exclusion column is provided, it is removed from the result.

    Parameters:
        df (pd.DataFrame): The original DataFrame.
        exclude (str): The name of a column to exclude from the numeric subset.

    Returns:
        pd.DataFrame: A new DataFrame with numeric columns, excluding the specified column (if any).
    """
    # Identify columns that are numeric types (int, float, etc.)
    numeric_columns = df.select_dtypes(include=["number"]).columns

    # If an exclusion column is specified and exists, remove it
    if exclude in numeric_columns:
        numeric_columns = numeric_columns.drop(exclude)

    # Select only the numeric columns (after exclusion)
    df_new = df[numeric_columns]

    # Return the new DataFrame
    return df_new

def analyze_correlations(df, target,method='pearson'):
    """
    Calculates and displays the Pearson correlation matrix for all numeric columns,
    and visualizes it using a heatmap for better interpretation.

    Parameters:
        - df (`pd.DataFrame`):
          The DataFrame containing the numeric columns for which correlations will be computed.

        - target (`str`):
          The name of the target column whose correlations with the others will be displayed.

        - method (`str`, default = `"pearson"`):
          The method used to calculate the correlation (e.g., 'pearson', 'spearman', or 'kendall').

    Returns:
        pd.DataFrame: The correlation matrix.
    """
    # Compute the correlation matrix
    correlation_matrix = df.corr(method=method)

    # Select only the row corresponding to the target column (as a Series)
    target_corr = correlation_matrix[target].drop(labels=[target])

    # Print the correlation matrix
    print(f"\nΠίνακας συσχετίσεων ({method})")
    print(target_corr)

    # Create a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(target_corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title(f"Heatmap Συσχετίσεων ({method})")
    plt.show()

    # Return the correlation matrix for further use
    return correlation_matrix

def plot_distributions(df):
    """
       Plots the distribution of all numeric columns in the given DataFrame.

       For each numeric feature, a histogram with a KDE (Kernel Density Estimate) line
       is displayed to visualize the data distribution, central tendency, and spread.

       Parameters:
           df (pd.DataFrame): Input DataFrame containing the numeric features.

       Returns:
           None: Displays histograms for each numeric column.
    """
    numeric_columns = df.select_dtypes(include=["number"]).columns
    for col in numeric_columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col], kde=True, bins=20)
        plt.title(f"Κατανομή: {col}")
        plt.xlabel(col)
        plt.ylabel("Πλήθος")
        plt.grid(True)
        plt.show()

def show_skew_kurtosis(df):
    """
        Calculates and prints skewness and kurtosis for all numeric columns in the DataFrame.

        Skewness measures the asymmetry of the distribution, while kurtosis measures
        the "tailedness" (sharpness of the peak) of the distribution.

        Parameters:
            df (pd.DataFrame): Input DataFrame containing numeric features.

        Returns:
            None: Prints skewness and kurtosis values for each numeric column.
        """
    numeric_columns = df.select_dtypes(include=["number"]).columns
    for col in numeric_columns:
        skew = df[col].skew()
        kurt = df[col].kurt()
        print(f"\n{col}")
        print(f"  Skewness (ασυμμετρία): {skew:.2f}")
        print(f"  Kurtosis (κυρτότητα): {kurt:.2f}")

def scale_features(df, method="standard"):
    """
    Scales/normalizes the data using one of several available scalers.

    Parameters:
        df (pd.DataFrame): DataFrame containing only numerical columns.
        method (str): Scaler selection:
            - `"standard"`: Standardization (mean = 0, std = 1)
            - `"minmax"`: Normalization to the range [0, 1]
            - `"maxabs"`: Scaling based on maximum absolute value ([-1, 1])
            - `"robust"`: Robust scaling (based on median and IQR, less sensitive to outliers)
            - `"normalizer"`: Normalizes each sample to a unit norm vector

    Returns:
        scaled_df: Scaled DataFrame with the same columns and index.
    """
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    elif method == "maxabs":
        scaler = MaxAbsScaler()
    elif method == "robust":
        scaler = RobustScaler()
    elif method == "normalizer":
        scaler = Normalizer()
    else:
        scaler = StandardScaler()
        print("Μη έγκυρη επιλογή method. Επιτρεπτές τιμές: 'standard','minmax','maxabs','robust','normalizer'. Επιλέχθηκε η default -> 'standard'")

    scaled_np = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_np, columns=df.columns, index=df.index)

    return scaled_df


def evaluate_kmeans_range(df, k_range=range(2, 11), init_method="k-means++", n_init=10, max_iter=300, tol=1e-4,
                          random_state=42):
    """
   Runs KMeans for a range of k values and displays Elbow & Silhouette Score plots.

    Parameters:
        df (array-like): Numerical data for clustering (NumPy array or DataFrame).
        k_range (range): Range of k values to evaluate. (default: 2–11)
        init_method (str): Initialization method ('k-means++', 'random', or array-like). (default: 'k-means++')
        n_init (int): Number of runs with different centroid seeds. (default: 10)
        max_iter (int): Maximum number of iterations per run. (default: 300)
        tol (float): Convergence threshold. (default: 1e-4)
        random_state (int): Random seed for reproducibility. (default: 42)

    Returns:
        None
    """
    # Lists for the Elbow plot
    elbow_k = []
    elbow_inertia = []

    # Lists for the Silhouette plot
    silhouette_k = []
    silhouette_scores = []

    for k in k_range:
        if k < 1:
            print(f"\nΤο k πρέπει να είναι ≥ 1. Παραλείπεται: k={k}")
            continue

        model = KMeans(
            n_clusters=k,
            init=init_method,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state
        )

        labels = model.fit_predict(df)

        # Logging for the Elbow plot
        elbow_k.append(k)
        elbow_inertia.append(model.inertia_)

        # Logging for the Silhouette only if k >= 2
        if k == 1:
            print("\nΤο Silhouette Score δεν ορίζεται για k=1.")
        else:
            score = silhouette_score(df, labels)
            silhouette_k.append(k)
            silhouette_scores.append(score)

    best_score = max(silhouette_scores)
    best_k = silhouette_k[silhouette_scores.index(best_score)]
    print(f"\nΒέλτιστο Silhouette Score: {best_score:.4f} για k = {best_k}")

    # Elbow plot
    plt.figure(figsize=(8, 6))
    plt.plot(elbow_k, elbow_inertia, marker='o', linestyle='--', color='b')
    plt.title("Elbow Method for Optimal K")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.xticks(elbow_k)
    plt.grid()
    plt.show()

    # Silhouette plot
    plt.figure(figsize=(8, 6))
    plt.plot(silhouette_k, silhouette_scores, marker='o', linestyle='--', color='g')
    plt.title("Silhouette Method for Optimal K")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.xticks(silhouette_k)
    plt.grid()
    plt.show()


def run_kmeans(df, n_clusters=8, init_method='k-means++', n_init=10, max_iter=300, tol=1e-4, random_state=42,
               compute_score=True):
    """
    Trains a KMeans model with the given data and returns the model and the labels.

    Parameters:
        df (array-like): Numerical data for clustering (NumPy array or DataFrame).
        n_clusters (int): Number of clusters.
        init_method (str): Initialization method ('k-means++', 'random', or array-like) (default: 'k-means++').
        n_init (int): Number of runs with different initial cluster centers (default: 10).
        max_iter (int): Maximum number of iterations for each run (default: 300).
        tol (float): Convergence tolerance threshold (default: 1e-4).
        random_state (int): Random seed for reproducibility (default: 42).
        compute_score (bool): If True, computes and prints the Silhouette Score for the fitted model (default: True).

    Returns:
        kmeans (KMeans): The trained model.
    """
    kmeans = KMeans(
        n_clusters=n_clusters,
        init=init_method,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state
    )

    kmeans.fit(df)
    # print("Cluster Centers:\n", kmeans.cluster_centers_)
    # print("Labels:\n", kmeans.labels_)
    print("Inertia:\n", kmeans.inertia_)

    # Compute Silhouette Score
    if compute_score and len(set(kmeans.labels_)) > 1:
        score = silhouette_score(df, kmeans.labels_)
        print(f"Silhouette Score: {score:.4f}")

    return kmeans


def plot_dendrogram_only(df, linkage_method='ward', metric='euclidean', figsize=(12, 6), title="Dendrogram",
                         truncate_lastp=None):
    """
    Creates a dendrogram for the given DataFrame using hierarchical clustering.

    Parameters
    ----------
    df : pd.DataFrame
        The input data (typically normalized numeric features).
    linkage_method : str
        Linkage method to use ('ward', 'complete', 'average', 'single').
    metric : str
        Distance metric ('euclidean', 'manhattan', etc.).
    figsize : tuple
        Figure size (width, height).
    title : str
        Plot title.
    truncate_lastp : int or None
        If provided, show only the last p merges (truncated dendrogram).

    Returns
    -------
    None
    """
    linked = scipy_linkage(df, method=linkage_method, metric=metric)

    plt.figure(figsize=figsize)

    if truncate_lastp:
        dendrogram(linked, no_labels=True, truncate_mode='lastp', p=truncate_lastp)
    else:
        dendrogram(linked, no_labels=True)

    plt.title(title)
    plt.xlabel("Δείγματα")
    plt.ylabel("Απόσταση συγχώνευσης")
    plt.grid(True)
    plt.show()

def run_agglomerative_clustering(
    df,
    n_clusters=2,
    metric='euclidean',
    compute_full_tree = 'auto',
    linkage = 'ward',
    distance_threshold = None,
    show_dendrogram = False,
    compute_score = True
    ):
    """
    Runs Agglomerative Clustering with optional dendrogram display and Silhouette scoring.

    Parameters
    ----------
    df : pd.DataFrame
        Normalized numeric input data.
    n_clusters : int, default=2
        Number of clusters (ignored if distance_threshold is provided).
    metric : str, default='euclidean'
        Distance metric ('euclidean', 'manhattan', etc.).
    compute_full_tree : {'auto', True, False}, default='auto'
        Whether to compute the full tree when using distance_threshold.
    linkage : str, default='ward'
        Linkage criterion ('ward', 'complete', 'average', 'single').
    distance_threshold : float or None, default=None
        Alternative to n_clusters: merge until this distance threshold.
    show_dendrogram : bool, default=False
        If True, display the dendrogram using scipy.
    compute_score : bool, default=True
        If True and there are >1 clusters, compute the Silhouette Score.

    Returns
    -------
    model : AgglomerativeClustering
        The fitted Agglomerative Clustering model.
    """
    labels = None

    # === Optional dendrogram ===
    if show_dendrogram:
        link = scipy_linkage(df, method=linkage, metric=metric)
        plt.figure(figsize=(12,6))
        dendrogram(link)
        plt.title("Dendrogram - Hierarchical Clustering")
        plt.xlabel("Δείγματα")
        plt.ylabel("Απόσταση συγχώνευσης")
        plt.grid(True)
        plt.show()

    # === Require either n_clusters or distance_threshold ===
    if n_clusters is None and distance_threshold is None:
        print("\nΠρέπει να δοθεί είτε n_clusters είτε distance_threshold.")
        return None

    # === Build the model ===
    model = AgglomerativeClustering(
        distance_threshold=distance_threshold,
        n_clusters=n_clusters,
        linkage=linkage,
        metric=metric,
        compute_full_tree=compute_full_tree,
        )

    # === Fit ===
    labels = model.fit_predict(df)

    # === Silhouette Score ===
    if compute_score and len(set(labels)) > 1:
        score = silhouette_score(df, labels)
        print(f"\nSilhouette Score: {score:.4f}")

    return model


def evaluate_agglomerative_range(
        df,
        k_range=range(2, 11),
        linkage='ward',
        metric='euclidean',
        show_scores=True
):
    """
    Runs Agglomerative Clustering for each k in k_range, computes the Silhouette Score,
    and optionally plots the scores.

    Parameters
    ----------
    df : pd.DataFrame or array-like
        Normalized numeric data for clustering.
    k_range : range, default=range(2, 11)
        Range of k values to evaluate.
    linkage : str, default='ward'
        Linkage type ('ward', 'complete', 'average', 'single').
    metric : str, default='euclidean'
        Distance metric ('euclidean', 'manhattan', etc.). If linkage='ward', this must be 'euclidean'.
    show_scores : bool, default=True
        If True, display a line plot of the Silhouette Scores.

    Returns
    -------
    None
    """
    silhouette_k = []  # k values evaluated
    silhouette_scores = []  # corresponding silhouette scores

    for k in k_range:
        if k < 2:
            print(f"\nΤο k πρέπει να είναι ≥ 2 για υπολογισμό Silhouette. Παραλείπεται: k={k}")
            continue

        # Create and fit the Agglomerative Clustering model
        model = AgglomerativeClustering(
            n_clusters=k,
            linkage=linkage,
            metric=metric
        )

        labels = model.fit_predict(df)

        # Compute Silhouette Score
        score = silhouette_score(df, labels)
        silhouette_k.append(k)
        silhouette_scores.append(score)

    # Find the best k based on the highest Silhouette Score
    best_score = max(silhouette_scores)
    best_k = silhouette_k[silhouette_scores.index(best_score)]
    print(f"\nΒέλτιστο Silhouette Score: {best_score:.4f} για k = {best_k}")

    # === Visualization ===
    if show_scores:
        plt.figure(figsize=(8, 6))
        plt.plot(silhouette_k, silhouette_scores, marker='o', linestyle='--', color='g')
        plt.title("Silhouette Method for Optimal K (Agglomerative)")
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Silhouette Score")
        plt.xticks(silhouette_k)
        plt.grid()
        plt.show()

    return None

def run_hdbscan_clustering(
    df,
    min_cluster_size=5,
    min_samples=None,
    cluster_selection_epsilon=0.0,
    metric='euclidean',
    alpha=1.0,
    cluster_selection_method='eom',
    show_probabilities=False,
    compute_score=False
    ):
    """
    Runs HDBSCAN clustering with optional Silhouette scoring and probability visualization.

    Parameters
    ----------
    df : pd.DataFrame
        Normalized numeric input data.
    min_cluster_size : int
        Minimum number of points per cluster.
    min_samples : int or None
        k for distance computation. If None, defaults to min_cluster_size.
    cluster_selection_epsilon : float
        Threshold for merging similar clusters.
    metric : str
        Distance metric ('euclidean', 'manhattan', etc.).
    alpha : float
        Parameter affecting clustering sensitivity.
    cluster_selection_method : str
        'eom' or 'leaf'.
    show_probabilities : bool
        If True, shows a histogram of per-point cluster membership probabilities.
    compute_score : bool
        If True and there is more than one label, compute the Silhouette Score.

    Returns
    -------
    model : HDBSCAN
        The fitted HDBSCAN model.
    """
    # === Build the model ===
    model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        metric=metric,
        alpha=alpha,
        cluster_selection_method=cluster_selection_method,
    )

    # === Fit ===
    labels = model.fit_predict(df)

    # === Silhouette Score (excluding outliers labeled -1) ===
    if compute_score:
        unique_labels = set(labels)

        if -1 in unique_labels:
            unique_labels.remove(-1) # remove outlier label to check if >1 clusters remain

        if len(unique_labels) > 1:
            mask = labels!=(-1) # mask points that are not outliers
            score = silhouette_score(df[mask],labels[mask])
            print(f"\nSilhouette Score (χωρίς τους outliers): {score:.4f}")
        else:
            print("\nSilhouette Score: Δεν υπολογίζεται (λιγότερα από 2 clusters).")

    # === Probability plot (optional)
    if show_probabilities:
        probs = model.probabilities_
        sns.histplot(probs, bins=20, kde=True)
        plt.title("Πιθανότητες συσχέτισης με το cluster (HDBSCAN)")
        plt.xlabel("Πιθανότητα")
        plt.ylabel("Πλήθος Δειγμάτων")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # === Report number of clusters and outliers ===
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_outliers = list(labels).count(-1)
    print(f"\nΑριθμός clusters: {n_clusters}")
    print(f"Αριθμός outliers: {n_outliers}")

    return model


def visualize_clusters(model, df, features, mode="2D"):
    """
    Visualizes clusters produced by KMeans, Agglomerative, or HDBSCAN in a 2D or 3D scatter plot.

    Parameters:
        model: Trained clustering model (KMeans, AgglomerativeClustering, HDBSCAN).
        df (pd.DataFrame): The dataset (with column names).
        features (list): A list of 2 or 3 column names for visualization.
        mode (str): '2D' or '3D'.

    Returns:
        None
    """
    if mode not in ["2D", "3D"]:
        print("\nΕπιτρεπτές τιμές για mode: '2D' ή '3D'")
        return

    for col in features:
        if col not in df.columns:
            print(f"\nΗ στήλη '{col}' δεν υπάρχει στο DataFrame.")
            return

    labels = model.labels_

    # If the model provides probabilities (e.g., HDBSCAN), use them for transparency
    if hasattr(model, 'probabilities_'):
        probs = model.probabilities_
    else:
        probs = np.ones(len(df))  # Without transparency

    centroids = getattr(model, 'cluster_centers_', None)  # Μόνο για KMeans

    if mode == '2D':
        x_col, y_col = features[0], features[1]
        plt.figure(figsize=(8, 6))

        # If outliers exist (label = -1), plot them separately
        outliers = labels == -1
        inliers = labels != -1

        # Regular data points
        plt.scatter(
            df[x_col],
            df[y_col],
            c=labels,
            cmap='viridis',
            s=50,
            alpha=probs  # Uses probabilities as transparency (HDBSCAN)
        )

        # Outliers (if any)
        if outliers.any():
            plt.scatter(
                df.loc[outliers, x_col],
                df.loc[outliers, y_col],
                c='black',
                marker='x',
                s=60,
                label='Outliers'
            )

        # Centroids (only for KMeans)
        if centroids is not None:
            plt.scatter(
                centroids[:, df.columns.get_loc(x_col)],
                centroids[:, df.columns.get_loc(y_col)],
                c='red',
                s=200,
                marker='X',
                label='Centroids'
            )

        if centroids is not None or outliers.any():
            plt.legend()

        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title("Cluster Visualization (2D)")
        plt.grid(True)
        plt.show()

    elif mode == '3D':
        if len(features) < 3:
            print("\nΓια προβολή 3D απαιτούνται 3 στήλες.")
            return

        x_col, y_col, z_col = features[0], features[1], features[2]
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        outliers = labels == -1
        inliers = labels != -1

        ax.scatter(
            df[x_col],
            df[y_col],
            df[z_col],
            c=labels,
            cmap='Set2',
            s=50,
            alpha=probs
        )

        if outliers.any():
            ax.scatter(
                df.loc[outliers, x_col],
                df.loc[outliers, y_col],
                df.loc[outliers, z_col],
                c='black',
                marker='x',
                s=60,
                label='Outliers'
            )

        if centroids is not None:
            ax.scatter(
                centroids[:, df.columns.get_loc(x_col)],
                centroids[:, df.columns.get_loc(y_col)],
                centroids[:, df.columns.get_loc(z_col)],
                c='red',
                s=200,
                marker='X',
                label='Centroids'
            )

        if centroids is not None or outliers.any():
            plt.legend()

        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_zlabel(z_col)
        plt.title("Cluster Visualization (3D)")
        plt.show()

def plot_cluster_results(df, models, titles, feature_pairs, figsize=(18, 5)):
    """
    Displays scatter plots for clustering results from different models,
    using feature pairs from the dataset.

    Each model is shown in its own subplot, where clustering labels are used
    as colors. If a model exposes the attribute `probabilities_`, it is used
    as point transparency (alpha).

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the data to visualize.
    models : list
        A list of trained clustering models with a `labels_` attribute.
    titles : list of str
        Titles to display above the corresponding plots.
    feature_pairs : list of tuple
        Feature pairs (e.g., [('sepal length', 'sepal width')]) to use on the x and y axes.
    figsize : tuple, optional
        Figure size (default: (18, 5)).

    Returns
    -------
    None
        The function renders the plots and does not return a value.
    """
    for pair in feature_pairs:
        fig, axes = plt.subplots(1, len(models), figsize=figsize)

        # If there is only one model, axes is not a list
        if len(models) == 1:
            axes = [axes]

        for i, (model, title) in enumerate(zip(models, titles)):
            ax = axes[i]

            labels = model.labels_
            if hasattr(model, 'probabilities_'):
                alpha = model.probabilities_
            else:
                alpha = 1.0

            ax.scatter(
                df[pair[0]],
                df[pair[1]],
                c=labels,
                cmap='Set2',
                s=50,
                alpha=alpha
            )
            ax.set_title(f"{title}\n{pair[0]} vs {pair[1]}")
            ax.set_xlabel(pair[0])
            ax.set_ylabel(pair[1])
            ax.grid(True)

        plt.tight_layout()
        plt.show()

def evaluate_clustering_models(true_labels_list, models, titles=None):
    """
    Computes and displays NMI, AMI, and ARI metrics for multiple clustering models.

    What it does:
    - Computes similarity metrics (NMI, AMI, ARI) between predicted and true labels
    - Prints each model’s metric values
    - Creates a bar chart to compare models across metrics

    Parameters
    ----------
    true_labels_list : list of array-like
        Ground-truth labels for each model.
    models : list
        List of clustering objects with a labels_ attribute.
    titles : list of str, optional
        Optional names for each model (e.g., 'KMeans', 'HDBSCAN').

    Returns
    -------
    None
        Displays results and a chart in the notebook.

    Example
    -------
    evaluate_clustering_models([y_true, y_true], [kmeans_model, hdbscan_model], titles=["KMeans", "HDBSCAN"])
    """
    if titles is None:
        titles = [f"Model {i+1}" for i in range(len(models))]

    metrics = ['NMI', 'AMI', 'ARI']
    scores = []

    for true_labels, model in zip(true_labels_list, models):
        if not hasattr(model, 'labels_'):
            print("Ένα από τα μοντέλα δεν έχει labels_. Το παραλείπουμε.")
            continue

        pred = model.labels_
        ari = adjusted_rand_score(true_labels, pred)
        nmi = normalized_mutual_info_score(true_labels, pred)
        ami = adjusted_mutual_info_score(true_labels, pred)
        scores.append([nmi, ami, ari])

        print(f"\n{titles[models.index(model)]}:")
        print(f"  NMI: {nmi:.4f}")
        print(f"  AMI: {ami:.4f}")
        print(f"  ARI: {ari:.4f}")

    # Convert to a numpy array for easier plotting
    scores = np.array(scores).T  # shape: (3 metrics, N models)

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, metric in enumerate(metrics):
        ax.bar(x + i*width, scores[i], width, label=metric)

    ax.set_xticks(x + width)
    ax.set_xticklabels(titles)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Clustering Evaluation Metrics per Model")
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

