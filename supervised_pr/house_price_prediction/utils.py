"""
    utils.py — Helper functions for the supervised regression project

    Purpose
    - Centralizes ALL reusable utilities for data loading/cleaning, EDA, feature prep,
      and model evaluation (so main.py stays focused on the execution flow).
    - Provides consistent, dataset-agnostic helpers that can be reused across projects.

    Contents (indicative)
    - I/O & Inspection:
        load_data, inspect_data, get_numeric_dataframe, clean_column_names
    - Cleaning & Transformation:
        clean_and_convert_column (e.g., to build target `price`)
    - EDA:
        plot_all_numeric_boxplots, print_outliers_for_all_numeric,
        plot_distributions, show_skew_kurtosis, analyze_correlations
    - Evaluation:
        cross_validated_rmse (returns per-fold RMSE/MAE and their means)

    Usage
      from utils import (
          load_data, inspect_data, get_numeric_dataframe, clean_column_names,
          clean_and_convert_column, analyze_correlations,
          plot_all_numeric_boxplots, print_outliers_for_all_numeric,
          plot_distributions, show_skew_kurtosis, cross_validated_rmse
      )

    Dependencies
    - pandas, numpy, matplotlib, seaborn, scikit-learn
    - Optionally: IPython.display.display (for DataFrame visualization)

    Notes
    - This module DEFINES functions only; no workflow is executed here.
    - Dataset paths, model choices, and plotting themes should be configured in main.py.
    - Functions aim to be side-effect-light and return explicit outputs where possible.

    Maintenance
    - Add new helpers (e.g., scalers, encoders, plotting utilities) here and import
      them explicitly in main.py.
    - Keep functions small, documented, and dataset-agnostic for easy reuse.
"""

# ===========================
# Utility functions
# ===========================

# --- Core libraries ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# --- sklearn preprocessing ---
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer
)

# --- ML models / evaluation ---
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    silhouette_score, normalized_mutual_info_score, adjusted_mutual_info_score,
    adjusted_rand_score, mean_squared_error, mean_absolute_error, r2_score,
    root_mean_squared_error
)


def load_data(source='file', filepath=None, dataset_func=None, sheet_name=None):
    """
    Loads data either from a file (.csv or .xls/.xlsx) or from a built-in sklearn dataset.

    Parameters:
        source (str): Source selection - 'file' or 'sklearn'. Default: 'file'.
        filepath (str): Full path of the file (CSV or Excel) if source='file'.
        dataset_func (function): Function from sklearn.datasets (e.g., load_iris) if source='sklearn'.
        sheet_name (str, optional): Sheet name when the file is Excel.

    Returns:
        tuple:
            - df (pd.DataFrame): Data as pandas DataFrame.
            - target (np.ndarray or None): Target labels when available (for sklearn), else None.
    """
    if source == 'file':
        if not filepath:
            print("\nΠαρακαλώ δώσε filepath για CSV.")
            return None, None
        try:
            if filepath.endswith('.csv'):
                df = pd.read_csv(filepath)
                print("\nDataset φορτώθηκε από CSV αρχείο:", df.shape)
                return df, None
            elif filepath.endswith(('.xls', '.xlsx')):
                if sheet_name is None:
                    # Show available sheets
                    xls = pd.ExcelFile(filepath)
                    print("\νΔιαθέσιμα φύλλα εργασίας (sheets):", xls.sheet_names)
                    print("Χρησιμοποίησε το όρισμα sheet_name για να διαλέξεις φύλλο.")
                    return None, None
                else:
                    df = pd.read_excel(filepath, sheet_name=sheet_name)
                    print("\n Dataset φορτώθηκε από XLS/XLSX αρχείο:", df.shape)
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
    - Data types & NaNs
    - Head
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


def get_numeric_features(df, exclude=None):
    """
    Returns the numeric columns of the DataFrame.

    Parameters:
        df (DataFrame): The dataset containing the columns.
        exclude (str): (Optional) Column name to remove from the numeric set.

    Returns:
        Index: Numeric columns without the exclude column (if provided).
    """
    # Find numeric dtype columns (int, float...)
    numeric_columns = df.select_dtypes(include=["number"]).columns

    # If an exclude column is given and exists, drop it
    if exclude in numeric_columns:
        numeric_columns = numeric_columns.drop(exclude)

    return numeric_columns


def plot_single_boxplot(column_data, column_name):
    """
    Shows a horizontal boxplot for one column and calculates IQR-based bounds.

    Parameters:
        column_data (Series): The series to visualize (e.g., df["Age"]).
        column_name (str): Column name used in the title.
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

    display(fig)
    plt.close(fig)

    print(f"\nΣτατιστικά για {column_name}")
    print(f"\nQ1 (25%): {Q1:.2f}")
    print(f"\nQ3 (75%): {Q3:.2f}")
    print(f"\nIQR: {IQR:.2f}")
    print(f"\nΠιθανοί outliers κάτω από: {lower_bound:.2f}")
    print(f"\nΠιθανοί outliers πάνω από: {upper_bound:.2f}")


def plot_all_numeric_boxplots(df, exclude=None):
    """
    Creates boxplots for every numeric column in the DataFrame,
    excluding the column specified by 'exclude' (e.g., 'CustomerID').

    Parameters:
        df (DataFrame): The dataset.
        exclude (str): (Optional) Column name to exclude.
    """
    # Get numeric columns and optionally exclude one
    numeric_columns = get_numeric_features(df, exclude)

    # Draw a boxplot and quantiles for each numeric column
    for col in numeric_columns:
        plot_single_boxplot(df[col], col)


def find_outliers_iqr(df, column):
    """
    Returns the rows of the DataFrame that are outliers
    in the given column according to the 1.5 * IQR rule.

    Parameters:
        df (DataFrame): The dataset.
        column (str): Column name to check.

    Returns:
        DataFrame: Rows identified as outliers.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - (1.5 * IQR)
    upper_bound = Q3 + (1.5 * IQR)

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers


def print_outliers_for_all_numeric(df, exclude=None):
    """
    Checks for outliers in all numeric columns (excluding any specified)
    and prints the outlier rows per column.

    Parameters:
        df (DataFrame): The dataset.
        exclude (str): (Optional) Column name to exclude, e.g., 'CustomerID'.
    """
    numeric_columns = df.select_dtypes(include=["number"]).columns
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
        columns (list): List of column names to check.

    Returns:
        DataFrame: New DataFrame without outliers.
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
    Returns a new DataFrame that contains only numeric columns.
    If an exclude column is provided and exists, it is removed.

    Parameters:
        df (pd.DataFrame): The original DataFrame.
        exclude (str): Column name to exclude.

    Returns:
        pd.DataFrame: New DataFrame with numeric columns, without the excluded column (if any).
    """
    # Select numeric columns (int, float...)
    numeric_columns = df.select_dtypes(include=["number"]).columns

    # Optionally drop an exclude column
    if exclude in numeric_columns:
        numeric_columns = numeric_columns.drop(exclude)

    df_new = df[numeric_columns]
    return df_new


def analyze_correlations(df, target=None, method="pearson"):
    """
    Computes and displays the correlation matrix for all numeric columns,
    with a heatmap for visualization. If a target column is provided,
    shows only correlations with the target and the respective heatmap.

    Parameters:
        df (pd.DataFrame): The DataFrame with numeric columns.
        target (str, optional): Target column name (e.g., for supervised learning).
                                If None, computes full correlations for all numerics.
        method (str): Correlation method. Allowed values: "pearson", "spearman", "kendall".

    Returns:
        pd.DataFrame: Correlation matrix (full or target-only).
    """
    correlation_matrix = df.corr(method=method)

    if target is not None:
        if target not in correlation_matrix.columns:
            raise ValueError(f"Η στήλη '{target}' δεν υπάρχει ή δεν είναι αριθμητική.")

        target_corr = correlation_matrix[target].drop(labels=[target])

        print(f"\nΠίνακας συσχετίσεων με τη μεταβλητή-στόχο '{target}' ({method})")
        print(target_corr)

        plt.figure(figsize=(8, 2))
        sns.heatmap(target_corr.to_frame().T, annot=True, cmap="coolwarm", fmt=".2f", square=True)
        plt.title(f"Heatmap Συσχετίσεων με '{target}'")
        plt.tight_layout()
        plt.show()

        return target_corr.to_frame()

    else:
        print(f"\nΠλήρης πίνακας συσχετίσεων ({method})")
        print(correlation_matrix)

        plt.figure(figsize=(10, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True)
        plt.title(f"Heatmap Συσχετίσεων ({method})")
        plt.tight_layout()
        plt.show()

        return correlation_matrix


def plot_distributions(df):
    """
    Plots a histogram with KDE for each numeric column in the DataFrame.
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
    Prints skewness and kurtosis for each numeric column in the DataFrame.
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
    Scales/normalizes the data using one of sklearn's scalers.

    Parameters:
        df (pd.DataFrame): DataFrame containing only numeric columns.
        method (str): Scaler choice:
            - "standard": Standardization (mean 0, std = 1)
            - "minmax": MinMax scaling to [0, 1]
            - "maxabs": Scaling by max absolute value ([-1, 1])
            - "robust": Robust scaling (median, IQR)
            - "normalizer": Normalizes each sample to unit norm

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


def clean_and_convert_column(series):
    """
    Cleans a string-like numeric column by removing unwanted characters and
    converts it to float. Returns the cleaned series and prints information.

    Parameters:
        series (pd.Series): Original column with values (possibly object/string).

    Returns:
        pd.Series: Cleaned numeric (float) column; failed conversions become NaN.
    """
    total = len(series)

    cleaned = (
        series
        .astype(str)
        .str.strip()
        .str.replace(',', '', regex=False)
        .str.replace(';', '', regex=False)
        .str.replace(';', '', regex=False)
        .str.replace('?', '', regex=False)
        .str.replace('€', '', regex=False)
        .str.replace(' ', '', regex=False)
        .pipe(pd.to_numeric, errors='coerce')
    )

    valid = cleaned.notna().sum()
    failed = total - valid

    print(f" Αρχικές τιμές: {total}")
    print(f" Επιτυχώς μετατράπηκαν: {valid}")
    print(f" Απέτυχαν και έγιναν NaN: {failed}")

    return cleaned


def clean_column_names(df):
    """
    Cleans DataFrame column names:
    - Strips leading/trailing spaces
    - Lowercases
    - Removes non-alphanumeric characters
    - Replaces internal whitespace with "_"

    Parameters:
        df (pd.DataFrame): DataFrame to clean.

    Returns:
        pd.DataFrame: Same DataFrame with cleaned column names.
    """
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r'[^\w\s]', '', regex=True)
        .str.replace(r'\s+', '_', regex=True)
    )
    print("\nΝέα ονόματα στηλών:")
    print(df.columns.to_list())
    return df


def cross_validated_rmse(model, X, y, cv=5, return_mean=False):
    """
    Computes RMSE and MAE via cross-validation using
    sklearn's 'neg_root_mean_squared_error' and 'neg_mean_absolute_error' scorers.

    Parameters:
        model: The scikit-learn model to evaluate.
        X (array-like): Features.
        y (array-like): Target variable.
        cv (int): Number of CV folds. Default: 5.
        return_mean (bool): If True, also returns mean RMSE and MAE.

    Returns:
        tuple:
            - If return_mean=False:
                (rmse_scores, mae_scores)
            - If return_mean=True:
                (rmse_scores, mean_rmse, mae_scores, mean_mae)
    """
    scores_rmse = cross_val_score(model, X, y, scoring="neg_root_mean_squared_error", cv=cv)
    rmse_scores = -scores_rmse

    scores_mae = cross_val_score(model, X, y, scoring="neg_mean_absolute_error", cv=cv)
    mae_scores = -scores_mae

    if return_mean:
        return rmse_scores, rmse_scores.mean(), mae_scores, mae_scores.mean()
    else:
        return rmse_scores, mae_scores