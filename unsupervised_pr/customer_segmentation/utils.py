"""
    utils.py — Helper functions for the clustering project

    Purpose
        - Centralizes ALL reusable functions (I/O, EDA, scaling,
          outlier detection, K-Means evaluation, visualizations).
        - Keeps the main execution “logic flow” in main.py for a clear separation of concerns.

    Contents (indicative)
        - load_data, inspect_data
        - plot_* (boxplots, 2D/3D scatter), analyze_correlations
        - find/remove/print outliers (IQR)
        - scale_features (Standard/MinMax/MaxAbs/Robust/Normalizer)
        - evaluate_kmeans_range (Elbow/Silhouette), run_kmeans, visualize_clusters
        - analyze_top_cluster

    Usage
          from utils import (
              load_data, inspect_data, plot_all_numeric_boxplots, ...,
              run_kmeans, visualize_clusters
          )

    Dependencies
        - pandas, numpy, matplotlib, seaborn, scikit-learn
        - Optionally IPython.display.display (used in plot_single_boxplot)

    Notes
        - The functions here DO NOT execute a “workflow”; they are only defined.
        - Global pandas display options should be set in main.py.
        - Dataset paths should be defined in main.py (e.g., DATA_PATH).

    Maintenance
        - When new functions are added, place them here and import them explicitly
          from main.py.
"""


# -------------------Imports for utilities---------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler,MaxAbsScaler, RobustScaler, Normalizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# For converting from Jupyter notebook to .py script
from IPython.display import display


def load_data(filepath):
    """
    Reads a CSV file and returns a DataFrame.

    Parameters:
        filepath (str): The path to the .csv file

    Returns:
        df (DataFrame): The data as a pandas DataFrame
    """
    df = pd.read_csv(filepath)
    print("\n->Dataset φορτώθηκε:", df.shape)
    return df


def inspect_data(df):
    """
    Displays basic information about the dataset:
    - Data types & NaN values
    - First few rows
    - Descriptive statistics
    """
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

    # Remove rows with the identified outlier indices
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


def analyze_correlations(df, target, method="pearson"):
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
    print(f"\nΠίνακας συσχετίσεων: ({method})")
    print(target_corr)

    # Create a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(target_corr.to_frame().T, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title("Heatmap Συσχετίσεων")
    plt.show()

    # Return the correlation matrix for further use
    return correlation_matrix


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
        print(
            "Μη έγκυρη επιλογή method. Επιτρεπτές τιμές: 'standard','minmax','maxabs','robust','normalizer'. Επιλέχθηκε η default -> 'standard'")

    scaled_np = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_np, columns=df.columns, index=df.index)

    return scaled_df


def evaluate_kmeans_range(X, k_range=range(2, 11), init_method="k-means++", n_init=10, max_iter=300, tol=1e-4,
                          random_state=42):
    """
    Runs KMeans for a range of k values and displays Elbow & Silhouette Score plots.

    Parameters:
        X (array-like): Numerical data for clustering (NumPy array or DataFrame).
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

        labels = model.fit_predict(X)

        # Logging for the Elbow plot
        elbow_k.append(k)
        elbow_inertia.append(model.inertia_)

        # Logging for the Silhouette only if k >= 2
        if k == 1:
            print("\nΤο Silhouette Score δεν ορίζεται για k=1.")
        else:
            score = silhouette_score(X, labels)
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


def run_kmeans(X, n_clusters=8, init_method='k-means++', n_init=10, max_iter=300, tol=1e-4, random_state=42):
    """
    Trains a KMeans model with the given data and returns the model and the labels.

    Parameters:
        X (array-like): Numerical data for clustering (NumPy array or DataFrame).
        n_clusters (int): Number of clusters.
        init_method (str): Initialization method ('k-means++', 'random', or array-like) (default: 'k-means++').
        n_init (int): Number of runs with different initial cluster centers (default: 10).
        max_iter (int): Maximum number of iterations for each run (default: 300).
        tol (float): Convergence tolerance threshold (default: 1e-4).
        random_state (int): Random seed for reproducibility (default: 42).

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

    kmeans.fit(X)
    # print("Cluster Centers:\n", kmeans.cluster_centers_)
    # print("Labels:\n", kmeans.labels_)
    print("Inertia:\n", kmeans.inertia_)
    return kmeans


def visualize_clusters(model, df, features, mode="2D"):
    """
    Visualizes the clusters produced by KMeans in a 2D or 3D scatter plot.

    Parameters:
        model (KMeans): Trained KMeans model.
        df (pd.DataFrame): The dataset (with column names).
        features (list): List of 2 or 3 column names for visualization.
        mode (str): '2D' or '3D'.

    Returns:
        None
    """
    # Validate the value of mode (2D or 3D)
    if mode not in ["2D", "3D"]:
        print("\nΕπιτρεπτές τιμές για mode: '2D' ή '3D'")
        return

    #  Check that all specified columns exist in the DataFrame
    for col in features:
        if col not in df.columns:
            print(f"\nΗ στήλη '{col}' δεν υπαρχει στο DataFrame.")
            return

    # Απόσπαση ετικετών cluster από το εκπαιδευμένο μοντέλο
    labels = model.labels_

    # Extract cluster labels from the trained model
    centroids = model.cluster_centers_

    # === 2D Visualization ===
    if mode == '2D':
        x_col, y_col = features[0], features[1]
        plt.figure(figsize=(8, 6))

        # Plot data points colored according to their cluster
        plt.scatter(
            df[x_col],
            df[y_col],
            c=labels,
            cmap='viridis',
            s=50
        )

        # Plot the cluster centers
        plt.scatter(
            centroids[:, df.columns.get_loc(x_col)],
            centroids[:, df.columns.get_loc(y_col)],
            c='red',
            s=200,
            marker='X',
            label='Centroids'
        )
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title("Cluster Visualization (2D)")
        plt.legend()
        plt.grid(True)
        plt.show()

    # === 3D Visualization ===
    elif mode == '3D':
        if len(features) < 3:
            print("\nΓια προβολή 2D απαιτούνται 3 στήλες.")
            return

        x_col, y_col, z_col = features[0], features[1], features[2]
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(df[x_col], df[y_col], df[z_col], c=labels, cmap='Set2', s=50)

        # Plot the centroids in 3D space
        ax.scatter(
            centroids[:, df.columns.get_loc(x_col)],
            centroids[:, df.columns.get_loc(y_col)],
            centroids[:, df.columns.get_loc(z_col)],
            c='red',
            s=200,
            marker='X',
            label='Centroids'
        )
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_zlabel(z_col)
        plt.title("Cluster Visualization (3D)")
        plt.legend()
        plt.show()


def analyze_top_cluster(model, df, target="Spending Score (1-100)", threshold=80, exclude=None):
    """
    Prints descriptive statistics (describe) for all clusters whose
    mean value in the target column is greater than or equal to the threshold.

    Parameters:
        model (KMeans): The trained KMeans model.
        df (pd.DataFrame): The DataFrame with the original (pre-scaling) data.
        target (str): The column used for evaluation (default: "Spending Score (1-100)").
        threshold (int or float): The mean-value threshold for target above which a cluster is considered interesting.
        exclude (str or list of str): Column name(s) to exclude (e.g., 'Gender_code').

    Returns:
        None: The function prints results and does not return a value.
    """
    df_working = df.copy()
    # If 'exclude' is provided, remove the specified columns
    if exclude:
        df_working = df_working.drop(exclude, axis=1, errors='ignore')

    # Add cluster labels from the model
    df_working["Cluster"] = model.labels_

    # Group by cluster and calculate the mean value of the target column
    summary = df_working.groupby("Cluster")[target].agg(["mean"]).sort_values(by="mean", ascending=False)

    # Select clusters that meet the threshold condition
    selected_clusters = []
    for index in summary.index:
        mean_value = summary.loc[index, "mean"]
        if mean_value >= threshold:
            selected_clusters.append(index)

    # If no suitable clusters are found, display a message and exit
    if not selected_clusters:
        print(f"Δεν βρέθηκε κανένα cluster με μέση τιμή {target} >= {threshold}")
        return

        # Display the selected clusters along with their mean target values
    print(f"Clusters με μέση τιμή {target} >= {threshold}:\n")
    print(summary.loc[selected_clusters])

    # Print descriptive statistics for each selected cluster
    for cluster_id in selected_clusters:
        filtered = df_working[df_working["Cluster"] == cluster_id]
        print(f"\nΠεριγραφικά στατιστικά για Cluster {cluster_id}:\n")
        print(filtered.describe())

