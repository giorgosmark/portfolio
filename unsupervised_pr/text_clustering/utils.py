"""
    utils.py — Helper functions for the text clustering project

    Purpose
        - Centralizes ALL reusable functions related to text preprocessing, embedding,
          dimensionality reduction, clustering, and evaluation.
        - Keeps the main execution logic separate (in main.py) for a clean and modular design.

    Contents (indicative)
        - download_text_model, load_data, inspect_data
        - clean_tokenize, token_vectorized
        - evaluate_kmeans_range, run_kmeans, run_agglomerative_clustering, run_hdbscan_clustering
        - evaluate_clustering_models, add_results
        - Visualization utilities: dendrograms, t-SNE, PCA variance plots, scatter plots

    Usage
          from utils import (
              load_data, clean_tokenize, token_vectorized,
              evaluate_kmeans_range, run_kmeans,
              run_agglomerative_clustering, run_hdbscan_clustering,
              evaluate_clustering_models
          )

    Dependencies
        - pandas, numpy, matplotlib, seaborn, scikit-learn, gensim, scipy
        - Optionally: IPython.display.display (used for DataFrame visualization)

    Notes
        - The functions here DO NOT perform an execution flow; they are meant to be reused
          from main.py.
        - Pretrained models (e.g., FastText) should be downloaded using download_text_model().
        - Datasets should be loaded and paths configured in main.py.

    Maintenance
        - When adding new functions, place them here and import explicitly from main.py.
        - Keep the file modular and dataset-agnostic, so it can support other text datasets
          (e.g., BBC News, 20 Newsgroups, etc.).
"""

# ===========================
#  Load required libraries
# ===========================

# --- Core libraries ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# used because some functions call display(fig) (project started in a notebook)
from IPython.display import display


# ===========================
#  Data preprocessing
# ===========================
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    MaxAbsScaler,
    RobustScaler,
    Normalizer
)

# ===========================
#  Machine Learning Models
# ===========================

#--- Gensim: download pretrained models or train our own ---
import gensim.downloader as api
from gensim.models import FastText

# --- Convert texts into a TF-IDF matrix ---
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Supervised learning (Regression) ---
from sklearn.linear_model import LinearRegression

# --- Unsupervised learning (Clustering) ---
from sklearn.cluster import KMeans, AgglomerativeClustering, HDBSCAN
#  --- Dendrograms for hierarchical clustering ---
from scipy.cluster.hierarchy import dendrogram, linkage as scipy_linkage

# ===========================
#  Model Evaluation
# ===========================

# --- Utilities for train/test split and cross-validation ---
from sklearn.model_selection import train_test_split, cross_val_score

# --- Evaluation metrics for clustering and regression ---
from sklearn.metrics import (
    silhouette_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    adjusted_rand_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error
)

# --- PCA & t-SNE for dimensionality reduction and visualization ---
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# --- Dataset: 20 Newsgroups (for text clustering experiments) ---
from sklearn.datasets import fetch_20newsgroups


def download_text_model(api_to_load):
    """
    Downloads and returns a pretrained text model from the Gensim API.

    Parameters
    ----------
    api_to_load : str
        The name of the model available in gensim.downloader (e.g. "fasttext-wiki-news-subwords-300")

    Returns
    -------
    pretrained_model : gensim model
        The downloaded pretrained model
    """
    print(f"\nDownloading pretrained model: {api_to_load} ...")
    print("This may take a few minutes depending on your internet speed.\n")

    pretrained_model = api.load(api_to_load)

    print("\nModel downloaded successfully.")
    return pretrained_model

def load_data(source='file', filepath=None,  dataset_func=None, sheet_name=None,):
    """
    Loads data either from a file (.csv or .xls/.xlsx) or from an sklearn built-in dataset.

    Parameters
    ----------
    source : str, default='file'
        Data source type - 'file' or 'sklearn'.
    filepath : str, optional
        Full path to the file (CSV or Excel) if source='file'.
    dataset_func : function, optional
        sklearn.datasets function (e.g., load_iris) if source='sklearn'.
    sheet_name : str, optional
        Name of the sheet if the file is an Excel workbook.

    Returns
    -------
    tuple
        - df (pd.DataFrame): The dataset as a pandas DataFrame.
        - target (np.ndarray or None): The target labels if available (for sklearn datasets), else None.
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
                    #Display available sheets
                    xls = pd.ExcelFile(filepath)
                    print("\nΔιαθέσιμα φύλλα εργασίας (sheets):", xls.sheet_names)
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
    - Data types and missing values
    - First rows preview
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


def clean_tokenize(text):
    """
    Performs basic text cleaning and tokenization.

    What it does:
    - Converts all characters to lowercase
    - Splits the text into words (tokens) based on spaces
    - Removes trailing periods (.) or commas (,) if they appear only once

    Parameters
    ----------
    text : str
        The input raw text.

    Returns
    -------
    clear_tokens : list of str
        A list of cleaned tokens with punctuation removed from the end.
    """
    text = text.lower()
    tokens = text.split()
    clear_tokens = []

    for word in tokens:

        if word.endswith(".") and word.count(".") == 1:
            word = word.rstrip(".")

        if word.endswith(",") and word.count(",") == 1:
            word = word.rstrip(",")

        clear_tokens.append(word)

    return clear_tokens

def token_vectorized(df, feature_name, model, handles_uw=False):
    """
    Computes document vectors from per-record tokens using a pretrained or custom
    Word2Vec/FastText-style model.

    What it does:
    - For each token list, averages the corresponding word embeddings
    - If the model doesn't handle unknown words, checks vocabulary membership first
    - If no valid token is found for a document, returns a zero vector

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing token lists.
    feature_name : str
        Name of the column that holds the token lists.
    model :
        The embedding model (e.g., gensim Word2Vec, FastText).
    handles_uw : bool, default=False
        Set True if the model can produce vectors for unknown words (e.g., custom FastText
        with subword support).

    Returns
    -------
    df_vectors : pd.DataFrame
        A DataFrame where each row is a document vector.

    Example
    -------
    vectors_df = token_vectorized(df, feature_name='tokens', model=fasttext_model)
    """
    # Final list of document vectors
    document_vectors = []

    # Iterate over each token list in the DataFrame
    for tokens in df[feature_name]:
        valid_vectors = [] # Temporary store for valid word vectors

        for word in tokens:
            if handles_uw:
                # If the model can handle unknown words, no vocab check needed
                valid_vectors.append(model.wv[word])
            else:
                # Otherwise, include the vector only if the word exists in the vocab
                if word in model:
                    valid_vectors.append(model[word])

        # Average all valid word vectors to get the document vector
        if valid_vectors:
            document_vector = np.mean(valid_vectors, axis=0)
        else:
            # If nothing valid was found, fall back to a zero vector
            document_vector = np.zeros(model.vector_size)

        document_vectors.append(document_vector)

    # Return as a DataFrame
    df_vectors = pd.DataFrame(document_vectors)
    return df_vectors


def evaluate_kmeans_range(df, k_range=range(2, 11), init_method="k-means++", n_init=10, max_iter=300, tol=1e-4,
                          random_state=42):
    """
    Runs KMeans for multiple k values and displays Elbow & Silhouette score plots.

    Parameters
    ----------
    df : array-like
        Numeric data for clustering (NumPy array or DataFrame).
    k_range : range, default=range(2, 12)
        Range of k values to evaluate.
    init_method : str or array-like, default='k-means++'
        Centroid initialization method ('k-means++', 'random', or array-like).
    n_init : int, default=10
        Number of initializations with different starting centroids.
    max_iter : int, default=300
        Maximum number of iterations per run.
    tol : float, default=1e-4
        Convergence tolerance.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
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

        # Record values for the Elbow plot
        elbow_k.append(k)
        elbow_inertia.append(model.inertia_)

        # Record values for Silhouette only if k >= 2
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
    Trains a KMeans model on the data and returns the fitted model.

    Parameters
    ----------
    df : array-like
        Numeric data for clustering (NumPy array or DataFrame).
    n_clusters : int
        Number of clusters.
    init_method : str or array-like, default='k-means++'
        Centroid initialization method ('k-means++', 'random', or array-like).
    n_init : int, default=10
        Number of runs with different centroid seeds.
    max_iter : int, default=300
        Maximum number of iterations for each run.
    tol : float, default=1e-4
        Convergence tolerance.
    random_state : int, default=42
        Random seed for reproducibility.
    compute_score : bool, default=True
        Whether to compute and print the Silhouette Score.

    Returns
    -------
    kmeans : KMeans
        The fitted KMeans model.
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

    # === Silhouette Score ===
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
    silhouette_k = []   # k values evaluated
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

    # ===  Require either n_clusters or distance_threshold ===
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

# --- Consolidated results table ---
clustering_results_df = pd.DataFrame(columns=[
    "Dataset", "Embedding", "Dim_Reduction", "Clustering", "n_clusters",
    "Silhouette", "NMI", "AMI", "ARI", "Notes"
])

def add_results(dataset, embedding, dim_reduction, clustering, n_clusters, silhouette, nmi, ami, ari, notes=""):
    """
    Adds a new clustering result entry to the global results table (clustering_results_df).

    What it does:
    - Creates a new dictionary entry with clustering results from one experiment
    - Appends it as a new row to the global DataFrame clustering_results_df

    Parameters
    ----------
    dataset : str
        Name of the dataset (e.g., "Iris").
    embedding : str
        Vectorization technique (e.g., "TF-IDF", "FastText").
    dim_reduction : str
        Dimensionality reduction technique (e.g., "PCA", "t-SNE", or "None").
    clustering : str
        Name of the clustering algorithm (e.g., "KMeans", "HDBSCAN").
    n_clusters : int or str
        Number of clusters (or "auto" for density-based methods).
    silhouette : float
        Silhouette Score for the clustering result.
    nmi : float
        Normalized Mutual Information score.
    ami : float
        Adjusted Mutual Information score.
    ari : float
        Adjusted Rand Index score.
    notes : str, optional
        Additional notes or remarks (e.g., observations, hyperparameters).

    Returns
    -------
    None
        Updates the global DataFrame clustering_results_df.
    """
    new_entry = {
        "Dataset": dataset,
        "Embedding": embedding,
        "Dim_Reduction": dim_reduction,
        "Clustering": clustering,
        "n_clusters": n_clusters,
        "Silhouette": silhouette,
        "NMI": nmi,
        "AMI": ami,
        "ARI": ari,
        "Notes": notes
    }
    global clustering_results_df
    clustering_results_df = pd.concat([clustering_results_df, pd.DataFrame([new_entry])], ignore_index=True)

