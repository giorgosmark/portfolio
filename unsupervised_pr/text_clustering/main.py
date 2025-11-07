"""
    main.py — Main workflow for the Text Clustering project

    Purpose:
        - Executes the complete text clustering pipeline step by step.
        - Utilizes helper functions from utils.py for data loading, preprocessing,
          embedding, dimensionality reduction, clustering, and evaluation.

    Workflow Steps:
          1. Load datasets (BBC News, 20 Newsgroups)
          2. Inspect datasets (shape, info, basic statistics)
          3. Text preprocessing (cleaning, tokenization)
          4. Vectorization using pretrained and custom FastText models
          5. Apply TF-IDF vectorization for comparison
          6. Visualize embeddings with t-SNE (2D projection)
          7. Apply PCA to analyze explained variance and reduce dimensionality (~90%)
          8. Evaluate K-Means across k ranges (Elbow + Silhouette)
          9. Train final clustering models:
               - K-Means
               - Agglomerative (Hierarchical)
               - HDBSCAN
         10. Visualize dendrograms for hierarchical clustering
         11. Evaluate all models using NMI, AMI, and ARI metrics
         12. Compare clustering results across embeddings and datasets
         13. Record all results in the summary table (clustering_results_df)

    Notes:
        - Paths and dataset configuration should be adjusted inside main.py.
        - This script provides the full experimental pipeline for reproducibility.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import display to show objects (e.g., DataFrames / figures)
# (Some functions still use display because the project was originally built in a notebook)
from IPython.display import display

# --- pandas display settings ---
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)

# Import all helper functions/variables from utils
from utils import (
    download_text_model,           # download or load a pre-trained model
    load_data,                     # load files or sklearn datasets
    inspect_data,                  # quick inspection of a DataFrame
    clean_tokenize,                # basic text cleaning and tokenization
    token_vectorized,              # convert tokens into document vectors
    evaluate_kmeans_range,         # k-means range evaluation for Elbow/Silhouette
    run_kmeans,                    # run k-means and print metrics
    plot_dendrogram_only,          # plot only the dendrogram (hierarchical)
    evaluate_agglomerative_range,  # evaluate range for Agglomerative clustering
    run_agglomerative_clustering,  # run Agglomerative clustering
    run_hdbscan_clustering,        # run HDBSCAN
    evaluate_clustering_models,    # compare models (NMI/AMI/ARI)
    add_results,                   # add results to the summary DataFrame
    clustering_results_df          # summary DataFrame for clustering results
)

#  --- Extra imports needed in the flow ---
from sklearn.decomposition import PCA  # PCA
from sklearn.manifold import TSNE  # t-SNE
from gensim.models import FastText  # FastText
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF
from sklearn.model_selection import train_test_split  # data splitting

# ===========================
#  Load pretrained FastText model & BBC dataset
# ===========================

# Downloads/loads the pretrained FastText model
pretrained_model_fasttext_model = download_text_model("fasttext-wiki-news-subwords-300")

# Loads the BBC dataset into a DataFrame along with target labels
df, target = load_data(source='file', filepath='datasets/bbc_news_test.csv')

# Quick check of basic DataFrame info (shape, dtypes, sample)
inspect_data(df)

# Create a token list for each article in the 'Text' column
df["Tokens"] = df["Text"].apply(clean_tokenize)

# Recheck structure after adding the Tokens column
inspect_data(df)

# Find words similar to "economy" using the model
similar = pretrained_model_fasttext_model.most_similar("economy")
# Print the closest words
print(similar)

# Transforms texts into vectors (average of words found in the model)
df_vectors = token_vectorized(df, "Tokens", pretrained_model_fasttext_model)

# Set up t-SNE for 2D projection
tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
# Compute the 2D coordinates
df_vectors_tsne = tsne.fit_transform(df_vectors)
# Create the figure
plt.figure(figsize=(8, 6))
# Plot the points
plt.scatter(df_vectors_tsne[:, 0], df_vectors_tsne[:, 1], s=10, alpha=0.6)
# Set the title
plt.title("t-SNE Pretrained Fasttext σε 2D")
# Enable grid
plt.grid(True)
# Show the plot
plt.show()

# Test KMeans for k=2..10 and show Inertia/Silhouette scores
evaluate_kmeans_range(df_vectors, k_range=range(2,11), init_method="k-means++", n_init=10, max_iter=300, tol=1e-4, random_state=42)

# Run KMeans with 2 clusters (optimal) and keep the model
kmeans_model_bbc = run_kmeans(df_vectors, n_clusters=2)

# Plot dendrogram for hierarchical clustering
plot_dendrogram_only(df_vectors, linkage_method='ward', metric='euclidean', figsize=(12, 6), title="Dendrogram", truncate_lastp=15)

# Evaluate Agglomerative for k=2..10 and show Silhouette score per k
evaluate_agglomerative_range(df_vectors, k_range=range(2, 11), linkage='ward', metric='euclidean', show_scores=True)

# Train Agglomerative with n_clusters=2 (based on evaluation)
agg_model_bbc = run_agglomerative_clustering(df_vectors, n_clusters=2, show_dendrogram=False)

# Run HDBSCAN on df_vectors with min_cluster_size=10 and min_samples=5
hdbscan_model_bbc = run_hdbscan_clustering(df_vectors, min_cluster_size=10, min_samples=5, compute_score=True, show_probabilities=False)

# Compare KMeans, Agglomerative, and HDBSCAN against true labels
models = [kmeans_model_bbc, agg_model_bbc, hdbscan_model_bbc]
# Titles for plots/prints
titles = ["KMeans (k = 2)", "Agglomerative (k = 2)", "HDBSCAN (k = 2)"]
# List of true labels for comparison
true_labels_list = [df["Category"], df["Category"], df["Category"]]
# Run NMI/AMI/ARI evaluation
evaluate_clustering_models(true_labels_list, models, titles)

# Log results into the summary DataFrame
add_results("BBC News dataset",
        "Pretrained Fasttext",
        "None",
        "KMeans",
        2,
        0.1539,
        0.1886,
        0.1876,
        0.1299,
        "Χαμηλά αποτελέσματα")

# Log results into the summary DataFrame
add_results("BBC News dataset",
        "Pretrained Fasttext",
        "None",
        "Agglomerative",
        2,
        0.1255,
        0.3001,
        0.2993,
        0.2329,
        "Λίγο καλύτερο από KMeans")

# Log results into the summary DataFrame
add_results("BBC News dataset",
        "Pretrained Fasttext",
        "None",
        "HDBSCAN",
        2,
        0.4357,
        0.0344,
        0.0316,
        0.0013,
        "Πολύ χαμηλά αποτελέσματα")

# ===========================
#  PCA on BBC (pretrained)
# ===========================

# Create PCA without specifying components to inspect explained variance ratio
pca_full = PCA()
# Fit PCA on the document vectors (pretrained FastText)
pca_full.fit(df_vectors)
# Compute cumulative explained variance to choose the number of components
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
# Set up the figure for the cumulative variance plot
plt.figure(figsize=(8, 5))
# Plot cumulative variance as a function of the number of components
plt.plot(cumulative_variance, marker='o')
# X-axis label (number of components)
plt.xlabel("Number of Components")
# Y-axis label (cumulative explained variance)
plt.ylabel("Cumulative Explained Variance")
# Set plot title
plt.title("PCA: Cumulative Variance Explained")
# Enable grid for readability
plt.grid(True)
# Horizontal reference line at 90% for visual guidance
plt.axhline(y=0.9, color='r', linestyle='--', label="90% Variance")
# Show legend
plt.legend()
# Display the plot
plt.show()

# Create PCA keeping ~65 components (based on the visual selection)
pca_65n = PCA(65)
# Project document vectors into the lower-dimensional (65D) space
df_vectors_pca = pca_65n.fit_transform(df_vectors)
# Set up 2D t-SNE on the PCA-reduced representations for visualization
tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
# Produce 2D embeddings from the PCA vectors
df_vectors_pca_tsne = tsne.fit_transform(df_vectors_pca)
# Create drawing canvas
plt.figure(figsize=(8, 6))
# Scatter plot of the 2D embeddings
plt.scatter(df_vectors_pca_tsne[:, 0], df_vectors_pca_tsne[:, 1], s=10, alpha=0.6)
# Set plot title
plt.title("t-SNE Pretrained Fasttext με PCA σε 2D")
# Enable grid for clarity
plt.grid(True)
# Show the plot
plt.show()

# Re-evaluate KMeans (on PCA vectors) for k=2..10
evaluate_kmeans_range(df_vectors_pca, k_range=range(2,11), init_method="k-means++", n_init=10, max_iter=300, tol=1e-4, random_state=42)

# Train KMeans with n_clusters=2 on the PCA vectors and keep the model
kmeans_model_bbc_pca65 = run_kmeans(df_vectors_pca, n_clusters=2)

# Evaluate Agglomerative for k=2..10 on the PCA vectors
evaluate_agglomerative_range(df_vectors_pca, k_range=range(2, 11), linkage='ward', metric='euclidean', show_scores=True)

# Train Agglomerative with n_clusters=2 on the PCA vectors
agg_model_bbc_pca65 = run_agglomerative_clustering(df_vectors_pca, n_clusters=2, show_dendrogram=False)

# Run HDBSCAN on the PCA vectors with the same base parameters
hdbscan_model_bbc_pca65 = run_hdbscan_clustering(df_vectors_pca, min_cluster_size=10, min_samples=5, compute_score=True, show_probabilities=False)


# Prepare the model list for post-PCA comparison
models = [kmeans_model_bbc_pca65, agg_model_bbc_pca65, hdbscan_model_bbc_pca65]
# Set titles for the corresponding results
titles = ["KMeans (k = 2)", "Agglomerative (k = 2)", "HDBSCAN"]
# Define ground-truth labels for each comparison
true_labels_list = [df["Category"], df["Category"], df["Category"]]
# Run NMI/AMI/ARI evaluation for the three models (with PCA)
evaluate_clustering_models(true_labels_list, models, titles)


# Record KMeans result (with PCA=65)
add_results("BBC News dataset",
        "Pretrained Fasttext",
        "PCA 65",
        "KMeans",
        2,
        0.1720,
        0.1886,
        0.1876,
        0.1299,
        "Χαμηλά αποτελέσματα")

# Record Agglomerative result (with PCA=65)
add_results("BBC News dataset",
        "Pretrained Fasttext",
        "PCA 65",
        "Agglomerative",
        2,
        0.1370,
        0.4122,
        0.4115,
        0.3016,
        "Αρκετή βελτίωση αλλά όχι αρκετή")

# Record HDBSCAN result (with PCA=65)
add_results("BBC News dataset",
        "Pretrained Fasttext",
        "PCA 65",
        "HDBSCAN",
        2,
        0.4751,
        0.0361,
        0.0332,
        0.0012,
        "Κακά αποτελέσματα")

# Train a custom FastText (CBOW: sg=0) on BBC tokens (same dimensionality as pretrained)
ft_custom_sg_0 = FastText(
    sentences=df["Tokens"],
    vector_size=300,       # keep 300 dims for compatibility
    window=5,              # left/right context window
    min_count=2,           # ignore very rare words
    sg=0,                  # CBOW
    epochs=10,             # training epochs
    workers=1              # single worker for reproducibility
)

# Try most_similar on an OOV term ("coolhead") with the trained FastText (CBOW) to check behavior
similar = ft_custom_sg_0.wv.most_similar("coolhead")
# Print the most similar words (no KeyError expected as with the pretrained model)
print(similar)

# Build document vectors with the custom FastText (CBOW), enabling handles_uw for OOV support
df_vectors_trainedsg0 = token_vectorized(df, "Tokens", ft_custom_sg_0, handles_uw=True)

# Set up 2D t-SNE for the new embeddings (custom FastText CBOW)
tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
# Produce 2D projections
df_vectors_trained_tsne = tsne.fit_transform(df_vectors_trainedsg0)
# Create canvas
plt.figure(figsize=(8, 6))
# Scatter the 2D points
plt.scatter(df_vectors_trained_tsne[:, 0], df_vectors_trained_tsne[:, 1], s=10, alpha=0.6)
# Set title
plt.title("t-SNE Custom Fasttext σε 2D")
# Enable grid
plt.grid(True)
# Show plot
plt.show()

# Evaluate KMeans range (k=2..10) on the custom embeddings (CBOW)
evaluate_kmeans_range(df_vectors_trainedsg0, k_range=range(2,11), init_method="k-means++", n_init=10, max_iter=300, tol=1e-4, random_state=42)

# Train KMeans with n_clusters=2 on the custom embeddings (CBOW)
kmeans_model_bbc_trained = run_kmeans(df_vectors_trainedsg0, n_clusters=2)

# Plot dendrogram (Ward) on the custom embeddings (CBOW)
plot_dendrogram_only(df_vectors_trainedsg0, linkage_method='ward', metric='euclidean', figsize=(12, 6), title="Dendrogram", truncate_lastp=15)

# Evaluate Agglomerative (k=2..10) on the custom embeddings (CBOW)
evaluate_agglomerative_range(df_vectors_trainedsg0, k_range=range(2, 11), linkage='ward', metric='euclidean', show_scores=True)

# Train Agglomerative with n_clusters=2 on the custom embeddings (CBOW)
agg_model_bbc_trained = run_agglomerative_clustering(df_vectors_trainedsg0, n_clusters=2, show_dendrogram=False)

# Run HDBSCAN (min_cluster_size=10, min_samples=5) on the custom embeddings (CBOW)
hdbscan_model_bbc_trained = run_hdbscan_clustering(df_vectors_trainedsg0, min_cluster_size=10, min_samples=5, compute_score=True, show_probabilities=False)

# Compare the three models (KMeans/Agglo/HDBSCAN) on the custom embeddings (CBOW)
models = [kmeans_model_bbc_trained, agg_model_bbc_trained, hdbscan_model_bbc_trained]
# Titles for the report
titles = ["KMeans (k = 2)", "Agglomerative (k = 2)", "HDBSCAN"]
# Define the true labels
true_labels_list = [df["Category"], df["Category"], df["Category"]]
# Run the evaluation
evaluate_clustering_models(true_labels_list, models, titles)

# Log KMeans result (custom CBOW, no PCA)
add_results("BBC News dataset",
        "Custom Fasttext SG=0",
        "None",
        "KMeans",
        2,
        0.2564,
        0.1614,
        0.1604,
        0.1187,
        "Χαμηλά αποτελέσματα")

# Log Agglomerative result (custom CBOW, no PCA)
add_results("BBC News dataset",
        "Custom Fasttext SG=0",
        "None",
        "Agglomerative",
        2,
        0.2903,
        0.1368,
        0.1356,
        0.0370,
        "Χαμηλά αποτελέσματα")

# Log HDBSCAN result (custom CBOW, no PCA)
add_results("BBC News dataset",
        "Custom Fasttext SG=0",
        "None",
        "HDBSCAN",
        3,
        0.2604,
        0.1633,
        0.1604,
        0.0586,
        "Χαμηλά αποτελέσματα")

# ===========================
#  PCA on Custom FastText (SG=0)
# ===========================

# Perform PCA on custom embeddings (CBOW) to inspect cumulative variance
pca_full = PCA()
# Fit PCA on df_vectors_trainedsg0
pca_full.fit(df_vectors_trainedsg0)

# Compute cumulative explained variance
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

# Plot cumulative variance
plt.figure(figsize=(8, 5))
# Cumulative variance curve
plt.plot(cumulative_variance, marker='o')
# X-axis label
plt.xlabel("Number of Components")
# Y-axis label
plt.ylabel("Cumulative Explained Variance")
# Chart title
plt.title("PCA: Cumulative Variance Explained (Custom FastText SG=0)")
# Enable grid
plt.grid(True)
# Horizontal line at 90%
plt.axhline(y=0.9, linestyle='--', label="90% Variance")
# Legend
plt.legend()
# Show figure
plt.show()


# Define PCA with 6 components (based on previous analysis)
pca_6n = PCA(6)
# Transform custom embeddings (CBOW) into the 6 most important PCA components
df_vectors_trained_pcasg0 = pca_6n.fit_transform(df_vectors_trainedsg0)

# Set up 2D t-SNE on PCA-reduced embeddings (6D)
tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
# Generate 2D projections
df_vectors_trained_tsnesg0 = tsne.fit_transform(df_vectors_trained_pcasg0)

# Create canvas
plt.figure(figsize=(8, 6))
# Scatter plot of the 2D points
plt.scatter(df_vectors_trained_tsnesg0[:, 0], df_vectors_trained_tsnesg0[:, 1], s=10, alpha=0.6)
# Set title
plt.title("t-SNE Custom FastText (SG=0) με PCA=6 σε 2D")
# Enable grid
plt.grid(True)
# Show plot
plt.show()


# Evaluate KMeans range (k=2..10) on PCA embeddings
evaluate_kmeans_range(df_vectors_trained_pcasg0, k_range=range(2,11), init_method="k-means++",
                      n_init=10, max_iter=300, tol=1e-4, random_state=42)

# Run KMeans with n_clusters=2 (optimal) on PCA embeddings
kmeans_model_bbc_trained_pcasg0 = run_kmeans(df_vectors_trained_pcasg0, n_clusters=2)

# Evaluate Agglomerative (k=2..10) on PCA embeddings
evaluate_agglomerative_range(df_vectors_trained_pcasg0, k_range=range(2, 11), linkage='ward',
                             metric='euclidean', show_scores=True)

# Run Agglomerative with n_clusters=5 (optimal) on PCA embeddings
agg_model_bbc_trained_pcasg0 = run_agglomerative_clustering(df_vectors_trained_pcasg0, n_clusters=5,
                                                            show_dendrogram=False)

# Run HDBSCAN (same base params) on PCA embeddings
hdbscan_model_bbc_trained_pcasg0 = run_hdbscan_clustering(df_vectors_trained_pcasg0, min_cluster_size=10,
                                                          min_samples=5, compute_score=True, show_probabilities=False)

# Compare the three models (KMeans/Agglo/HDBSCAN) after PCA=6
models = [kmeans_model_bbc_trained_pcasg0, agg_model_bbc_trained_pcasg0, hdbscan_model_bbc_trained_pcasg0]
# Titles for reporting
titles = ["KMeans (k = 2)", "Agglomerative (k = 5)", "HDBSCAN (k = 2)"]
# Define true labels
true_labels_list = [df["Category"], df["Category"], df["Category"]]
# Run evaluation
evaluate_clustering_models(true_labels_list, models, titles)

# Log KMeans result (custom CBOW, PCA=6)
add_results("BBC News dataset",
        "Custom Fasttext SG=0",
        "PCA 6",
        "KMeans",
        2,
        0.2880,
        0.1614,
        0.1604,
        0.1187,
        "Πολύ χαμηλά αποτελέσματα")

# Log Agglomerative result (custom CBOW, PCA=6)
add_results("BBC News dataset",
        "Custom Fasttext SG=0",
        "PCA 6",
        "Agglomerative",
        5,
        0.2580,
        0.4898,
        0.4880,
        0.4485,
        "Βελτιωμένα αποτελέσματα")

# Log HDBSCAN result (custom CBOW, PCA=6)
add_results("BBC News dataset",
        "Custom Fasttext SG=0",
        "PCA 6",
        "HDBSCAN",
        2,
        0.1834,
        0.0253,
        0.0227,
        0.0084,
        "Κακά αποτελέσματα")

# ===========================
#  Custom FastText (SG=1) on BBC
# ===========================

# rain a custom FastText model (Skip-gram: sg=1) on BBC tokens
ft_custom_sg_1 = FastText(
    sentences=df["Tokens"],   # list of tokens per article
    vector_size=300,          # embedding dimensionality
    window=5,                 # context window
    min_count=2,              # filter rare words
    sg=1,                     # Skip-gram
    epochs=10,                # training epochs
    workers=1                 # single worker for reproducibility
)

# Build document vectors with the custom FastText (SG=1), with OOV support enabled
df_vectors_trainedsg1 = token_vectorized(df, "Tokens", ft_custom_sg_1, handles_uw=True)

# Set up 2D t-SNE for the new embeddings (SG=1)
tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
# Generate 2D projections
df_vectors_trained_sg1_tsne = tsne.fit_transform(df_vectors_trainedsg1)

# Create canvas
plt.figure(figsize=(8, 6))
# Scatter plot of the 2D points
plt.scatter(df_vectors_trained_sg1_tsne[:, 0], df_vectors_trained_sg1_tsne[:, 1], s=10, alpha=0.6)
# Set title
plt.title("t-SNE Custom FastText (SG=1) σε 2D")
# Enable grid
plt.grid(True)
# Show plot
plt.show()

# Evaluate KMeans range (k=2..10) on the custom embeddings (SG=1)
evaluate_kmeans_range(df_vectors_trainedsg1, k_range=range(2,11), init_method="k-means++",
                      n_init=10, max_iter=300, tol=1e-4, random_state=42)

# Run KMeans with n_clusters=5 (matches BBC categories)
kmeans_model_bbc_trained_g1 = run_kmeans(df_vectors_trainedsg1, n_clusters=5)

# Evaluate Agglomerative (k=2..10) on the custom embeddings (SG=1)
evaluate_agglomerative_range(df_vectors_trainedsg1, k_range=range(2, 11), linkage='ward',
                             metric='euclidean', show_scores=True)

# Run Agglomerative with n_clusters=2 (based on best scores observed)
agg_model_bbc_trained_g1 = run_agglomerative_clustering(df_vectors_trainedsg1, n_clusters=2,
                                                        show_dendrogram=False)

# Run HDBSCAN (min_cluster_size=10, min_samples=5) on the SG=1 embeddings
hdbscan_model_bbc_trained_g1 = run_hdbscan_clustering(df_vectors_trainedsg1, min_cluster_size=10,
                                                      min_samples=5, compute_score=True, show_probabilities=False)

# Compare the three models (KMeans/Agglo/HDBSCAN) on the custom SG=1 embeddings
models = [kmeans_model_bbc_trained_g1, agg_model_bbc_trained_g1, hdbscan_model_bbc_trained_g1]
# Titles for the report
titles = ["KMeans (k = 5)", "Agglomerative (k = 2)", "HDBSCAN (k = 2)"]
# Define the true labels
true_labels_list = [df["Category"], df["Category"], df["Category"]]
# Run the evaluation
evaluate_clustering_models(true_labels_list, models, titles)

# Log KMeans result (custom SG=1, no PCA)
add_results("BBC News dataset",
        "Custom Fasttext SG=1",
        "None",
        "KMeans",
        5,
        0.2159,
        0.7782,
        0.7775,
        0.8055,
        "Πολύ καλά αποτελέσματα")

# Log Agglomerative result (custom SG=1, no PCA)
add_results("BBC News dataset",
        "Custom Fasttext SG=1",
        "None",
        "Agglomerative",
        2,
        0.2095,
        0.4857,
        0.4851,
        0.3283,
        "Μέτρια αποτελέσματα")

# Log HDBSCAN result (custom SG=1, no PCA)
add_results("BBC News dataset",
        "Custom Fasttext SG=1",
        "None",
        "HDBSCAN",
        2,
        0.1169,
        0.0152,
        0.0124,
        0.0012,
        "Πολύ χαμηλά αποτελέσματα")

# ===========================
#  PCA on Custom FastText (SG=1)
# ===========================

# Perform PCA on the custom embeddings (SG=1) to inspect cumulative variance
pca_full = PCA()
# Fit PCA on df_vectors_trainedsg1
pca_full.fit(df_vectors_trainedsg1)

# Compute cumulative explained variance
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

# Plot cumulative variance
plt.figure(figsize=(8, 5))
# Cumulative variance curve
plt.plot(cumulative_variance, marker='o')
# X-axis label
plt.xlabel("Number of Components")
# Y-axis label
plt.ylabel("Cumulative Explained Variance")
# Chart title
plt.title("PCA: Cumulative Variance Explained (Custom FastText SG=1)")
# Enable grid
plt.grid(True)
# Horizontal line at 90%
plt.axhline(y=0.9, linestyle='--', label="90% Variance")
# Legend
plt.legend()
# Show figure
plt.show()

# Define PCA with 20 components (close to ~90% variance)
pca_20n = PCA(20)
# Project the SG=1 embeddings onto the top 20 PCA components
df_vectors_trained_pcasg1 = pca_20n.fit_transform(df_vectors_trainedsg1)

# Set up 2D t-SNE on the PCA embeddings
tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
# Generate 2D projections
df_vectors_trained_pcasg1_tsne = tsne.fit_transform(df_vectors_trained_pcasg1)

# Create canvas
plt.figure(figsize=(8, 6))
# Scatter plot of the 2D points
plt.scatter(df_vectors_trained_pcasg1_tsne[:, 0], df_vectors_trained_pcasg1_tsne[:, 1], s=10, alpha=0.6)
# Set title
plt.title("t-SNE Custom FastText (SG=1) με PCA=20 σε 2D")
# Enable grid
plt.grid(True)
# Show plot
plt.show()

# Evaluate KMeans range (k=2..10) on the PCA embeddings (20 components)
evaluate_kmeans_range(df_vectors_trained_pcasg1, k_range=range(2,11), init_method="k-means++",
                      n_init=10, max_iter=300, tol=1e-4, random_state=42)

# Run KMeans with n_clusters=5 on the PCA embeddings (20 components)
kmeans_model_bbc_trained_pca20 = run_kmeans(df_vectors_trained_pcasg1, n_clusters=5)

# Plot dendrogram (Ward) on the PCA embeddings (20 components)
plot_dendrogram_only(df_vectors_trained_pcasg1, linkage_method='ward', metric='euclidean',
                     figsize=(12, 6), title="Dendrogram", truncate_lastp=15)

# Evaluate Agglomerative (k=2..10) on the PCA embeddings (20 components)
evaluate_agglomerative_range(df_vectors_trained_pcasg1, k_range=range(2, 11), linkage='ward',
                             metric='euclidean', show_scores=True)

# Run Agglomerative with n_clusters=5 on the PCA embeddings (20 components)
agg_model_bbc_trained_pca20 = run_agglomerative_clustering(df_vectors_trained_pcasg1, n_clusters=5,
                                                           show_dendrogram=False)

# Run HDBSCAN (min_cluster_size=10, min_samples=5) on the PCA embeddings (20 components)
hdbscan_model_bbc_trained_pca20 = run_hdbscan_clustering(df_vectors_trained_pcasg1, min_cluster_size=10,
                                                         min_samples=5, compute_score=True, show_probabilities=False)

# (Optional) Second HDBSCAN run with different parameters for comparison
hdbscan_model_bbc_trained_pca20_2 = run_hdbscan_clustering(df_vectors_trained_pcasg1, min_cluster_size=18,
                                                           min_samples=7, compute_score=True, show_probabilities=False)

# Compare the three main models (KMeans/Agglo/HDBSCAN-1) after PCA=20
models = [kmeans_model_bbc_trained_pca20, agg_model_bbc_trained_pca20, hdbscan_model_bbc_trained_pca20]
# Titles for the report
titles = ["KMeans (k = 5)", "Agglomerative (k = 5)", "HDBSCAN (k = 8)"]
# Define the true labels
true_labels_list = [df["Category"], df["Category"], df["Category"]]
# Run the evaluation
evaluate_clustering_models(true_labels_list, models, titles)

# Log KMeans result (custom SG=1, PCA=20)
add_results("BBC News dataset",
        "Custom Fasttext SG=1",
        "PCA 20",
        "KMeans",
        5,
        0.2612,
        0.7789,
        0.7782,
        0.8054,
        "Πολύ καλά αποτελέσματα")

# Log Agglomerative result (custom SG=1, PCA=20)
add_results("BBC News dataset",
        "Custom Fasttext SG=1",
        "PCA 20",
        "Agglomerative",
        5,
        0.2428,
        0.8106,
        0.8100,
        0.8359,
        "Πολύ καλά αποτελέσματα")

# Log HDBSCAN result (custom SG=1, PCA=20)
add_results("BBC News dataset",
        "Custom Fasttext SG=1",
        "PCA 20",
        "HDBSCAN",
        8,
        0.2906,
        0.3346,
        0.3295,
        0.0644,
        "Πολύ χαμηλά αποτελέσματα")

# Download the 20 Newsgroups dataset (without headers/footers/quotes)
data_20newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
# Extract the raw text documents
documents_20newsgroups = data_20newsgroups.data
# Extract numeric category labels
labels_20newsgroups = data_20newsgroups.target
# Extract category names
labels_names_20newsgroups = data_20newsgroups.target_names
# Build a dictionary with labels and text
news_dict = {'label': labels_20newsgroups,
             'comment': documents_20newsgroups}
# Convert it to a DataFrame
news_df = pd.DataFrame(news_dict)
# Add the readable category name for each label
news_df['label_name'] = news_df['label'].apply(lambda x: labels_names_20newsgroups[x])


# Split off a 10% test set with stratification
_, x_test_news, _, y_test_news = train_test_split(
    news_df,
    news_df['label'],
    test_size=0.1,
    random_state=42,
    stratify=news_df['label']
)

# Inspect basic stats/shape
inspect_data(x_test_news)

# Tokenize with the same light cleaning
x_test_news["Tokens"] = x_test_news["comment"].apply(clean_tokenize)

# Print tokens for a quick spot-check
print(x_test_news["Tokens"])

# Build document vectors with the pretrained FastText (Word2Vec-like behavior)
df_vectors_newsg = token_vectorized(x_test_news, "Tokens", pretrained_model_fasttext_model)

# Set up 2D t-SNE on the embeddings
tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
# Compute 2D projections
df_vectors_newsg_tsne = tsne.fit_transform(df_vectors_newsg)

# Create a canvas for the scatter plot
plt.figure(figsize=(8, 6))
# Plot the points
plt.scatter(df_vectors_newsg_tsne[:, 0], df_vectors_newsg_tsne[:, 1], s=10, alpha=0.6)
# Set title
plt.title("t-SNE Pretrained FastText (20NG) σε 2D")
# Enable grid
plt.grid(True)
# Show
plt.show()

# Evaluate KMeans for k=2..24
evaluate_kmeans_range(df_vectors_newsg, k_range=range(2,25), init_method="k-means++",
                      n_init=10, max_iter=300, tol=1e-4, random_state=42)

# Run KMeans with n_clusters=2 (best silhouette)
kmeans_model_newsg = run_kmeans(df_vectors_newsg, n_clusters=2)

# Plot a quick dendrogram (Ward) for a visual check
plot_dendrogram_only(df_vectors_newsg, linkage_method='ward', metric='euclidean',
                     figsize=(12, 6), title="Dendrogram (20NG, Pretrained) ", truncate_lastp=15)

# Evaluate Agglomerative for k=2..24
evaluate_agglomerative_range(df_vectors_newsg, k_range=range(2, 25), linkage='ward',
                             metric='euclidean', show_scores=True)

# Run Agglomerative with n_clusters=2
agg_model_newsg = run_agglomerative_clustering(df_vectors_newsg, n_clusters=2, show_dendrogram=False)

# Run HDBSCAN with basic parameters
hdbscan_model_newsg = run_hdbscan_clustering(df_vectors_newsg, min_cluster_size=10, min_samples=5,
                                             compute_score=True, show_probabilities=False)

# Compare KMeans/Agglomerative/HDBSCAN on 20NG (pretrained)
models = [kmeans_model_newsg, agg_model_newsg, hdbscan_model_newsg]
# Titles
titles = ["KMeans (k = 2)", "Agglomerative (k = 2)", "HDBSCAN (k = 2)"]
# True labels
true_labels_list = [x_test_news["label"], x_test_news["label"], x_test_news["label"]]
# Run evaluation
evaluate_clustering_models(true_labels_list, models, titles)

# Log KMeans results (pretrained, no PCA)
add_results("20 Newsgroups",
        "Pretrained Fasttext",
        "None",
        "KMeans",
        2,
        0.6072,
        0.0036,
        0.0002,
        -0.0001,
        "Κακά αποτελέσματα")

# Log Agglomerative results (pretrained, no PCA)
add_results("20 Newsgroups",
        "Pretrained Fasttext",
        "None",
        "Agglomerative",
        2,
        0.5814,
        0.0034,
        -0.0001,
        -0.0001,
        "Κακά αποτελέσματα")

# Log HDBSCAN results (pretrained, no PCA)
add_results("20 Newsgroups",
        "Pretrained Fasttext",
        "None",
        "HDBSCAN",
        2,
        0.5996,
        0.0066,
        -0.0005,
        -0.0002,
        "Κακά αποτελέσματα")

# Train PCA on the pretrained embeddings to inspect cumulative variance
pca_full = PCA()
# Fit on df_vectors_newsg
pca_full.fit(df_vectors_newsg)

# Υπολογίζω σωρευτική εξηγούμενη διασπορά
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

# Compute cumulative explained variance
plt.figure(figsize=(8, 5))
# Curve
plt.plot(cumulative_variance, marker='o')
# X label
plt.xlabel("Number of Components")
# Y label
plt.ylabel("Cumulative Explained Variance")
# Title
plt.title("PCA: Cumulative Variance Explained (20NG, Pretrained)")
# Grid
plt.grid(True)
# Reference line at 90%
plt.axhline(y=0.9, linestyle='--', label="90% Variance")
# Legend
plt.legend()
# Show
plt.show()

# Define PCA with 55 components (≈90% variance as agreed)
pca_55n = PCA(55)
# Transform embeddings with pca_55n
df_vectors_news20_pca = pca_55n.fit_transform(df_vectors_newsg)

# Set up 2D t-SNE on the PCA-reduced embeddings (55 components)
tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
# Υπολογίζω 2D προβολές# Compute 2D projections
df_vectors_newsg1_pca_tsne = tsne.fit_transform(df_vectors_news20_pca)

# Scatter plot
plt.figure(figsize=(8, 6))
# t-SNE points
plt.scatter(df_vectors_newsg1_pca_tsne[:, 0], df_vectors_newsg1_pca_tsne[:, 1], s=10, alpha=0.6)
# Title
plt.title("t-SNE Pretrained FastText με PCA=55 σε 2D (20NG)")
# Grid
plt.grid(True)
# Show
plt.show()

# Evaluate KMeans (k=2..24) on PCA=55 embeddings
evaluate_kmeans_range(df_vectors_news20_pca, k_range=range(2,25), init_method="k-means++",
                      n_init=10, max_iter=300, tol=1e-4, random_state=42)

# Run KMeans with n_clusters=2 on PCA=55 embeddings
kmeans_model_newsg_pca = run_kmeans(df_vectors_news20_pca, n_clusters=2)

# Plot dendrogram on PCA-reduced embeddings
plot_dendrogram_only(df_vectors_news20_pca, linkage_method='ward', metric='euclidean',
                     figsize=(12, 6), title="Dendrogram (20NG, Pretrained+PCA=55)", truncate_lastp=15)

# Evaluate Agglomerative (k=2..24) on PCA=55 embeddings
evaluate_agglomerative_range(df_vectors_news20_pca, k_range=range(2, 25), linkage='ward',
                             metric='euclidean', show_scores=True)

# Run Agglomerative with n_clusters=2 on PCA=55 embeddings
agg_model_newsg_pca = run_agglomerative_clustering(df_vectors_news20_pca, n_clusters=2, show_dendrogram=False)

# Run HDBSCAN on PCA=55 embeddings
hdbscan_model_newsg_pca = run_hdbscan_clustering(df_vectors_news20_pca, min_cluster_size=10, min_samples=5,
                                                 compute_score=True, show_probabilities=False)

# Compare the three models after PCA=55
models = [kmeans_model_newsg_pca, agg_model_newsg_pca, hdbscan_model_newsg_pca]
# Titles
titles = ["KMeans (k = 2)", "Agglomerative (k = 2)", "HDBSCAN"]
# True labels
true_labels_list = [x_test_news["label"], x_test_news["label"], x_test_news["label"]]
# Run evaluation
evaluate_clustering_models(true_labels_list, models, titles)

# Log KMeans results (pretrained, PCA=55)
add_results("20 Newsgroups",
        "Pretrained Fasttext με PCA",
        "PCA 55",
        "KMeans",
        2,
        0.6635,
        0.0036,
        0.0002,
        -0.0001,
        "Κακά αποτελέσματα")

# Log Agglomerative results (pretrained, PCA=55)
add_results("20 Newsgroups",
        "Pretrained Fasttext με PCA",
        "PCA 55",
        "Agglomerative",
        2,
        0.6480,
        0.0034,
        -0.0001,
        -0.0001,
        "Κακά αποτελέσματα")

# Log HDBSCAN results (pretrained, PCA=55)
add_results("20 Newsgroups",
        "Pretrained Fasttext με PCA",
        "PCA 55",
        "HDBSCAN",
        2,
        0.6623,
        0.0064,
        -0.0007,
        -0.0001,
        "Κακά αποτελέσματα")

# Train a custom FastText model (sg=1, larger window) on 20NG test-set tokens
ft_custom_2 = FastText(
    sentences=x_test_news["Tokens"],  # λίστες λέξεων
    vector_size=300,                  # διάσταση embeddings
    window=10,                        # ευρύτερα συμφραζόμενα
    min_count=2,                      # φίλτρο σπανίων
    sg=1,                             # Skip-gram
    epochs=10,                        # εποχές
    workers=1                         # αναπαραγωγιμότητα
)

# Convert documents to vectors using the custom FastText model
df_vectors_newsg_trained = token_vectorized(x_test_news, "Tokens", ft_custom_2, handles_uw=True)

# Set up 2D t-SNE on the custom embeddings
tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
# Compute 2D projections
df_vectors_newsg_trained_tsne = tsne.fit_transform(df_vectors_newsg_trained)

# Plot the t-SNE scatter
plt.figure(figsize=(8, 6))
# Points
plt.scatter(df_vectors_newsg_trained_tsne[:, 0], df_vectors_newsg_trained_tsne[:, 1], s=10, alpha=0.6)
# Title
plt.title("t-SNE Custom FastText (20NG, SG=1) σε 2D")
# Grid
plt.grid(True)
# Show
plt.show()

# Evaluate KMeans (k=2..24) on the custom embeddings
evaluate_kmeans_range(df_vectors_newsg_trained, k_range=range(2,25), init_method="k-means++",
                      n_init=10, max_iter=300, tol=1e-4, random_state=42)

# Run KMeans with n_clusters=3 (based on evaluation)
kmeans_model_newsg_trained = run_kmeans(df_vectors_newsg_trained, n_clusters=3)

# Evaluate Agglomerative (k=2..24) on the custom embeddings
evaluate_agglomerative_range(df_vectors_newsg_trained, k_range=range(2, 25), linkage='ward',
                             metric='euclidean', show_scores=True)

# Run Agglomerative with n_clusters=3
agg_model_newsg_trained = run_agglomerative_clustering(df_vectors_newsg_trained, n_clusters=3, show_dendrogram=False)

# Run HDBSCAN on the custom embeddings
hdbscan_model_newsg_trained = run_hdbscan_clustering(df_vectors_newsg_trained, min_cluster_size=10, min_samples=5,
                                                     compute_score=True, show_probabilities=False)

# Compare KMeans/Agglomerative/HDBSCAN on the custom embeddings
models = [kmeans_model_newsg_trained, agg_model_newsg_trained, hdbscan_model_newsg_trained]
# Titles
titles = ["KMeans (k = 3)", "Agglomerative (k = 3)", "HDBSCAN (k = 2)"]
# Labels for evaluation
true_labels_list = [x_test_news["label"], x_test_news["label"], x_test_news["label"]]
# Run evaluation
evaluate_clustering_models(true_labels_list, models, titles)

# Log KMeans results (custom, no PCA)
add_results("20 Newsgroups",
        "Custom Fasttext",
        "None",
        "KMeans",
        3,
        0.5674,
        0.0048,
        0.0002,
        -0.0001,
        "Κακά αποτελέσματα")

# Log Agglomerative results (custom, no PCA)
add_results("20 Newsgroups",
        "Custom Fasttext",
        "None",
        "Agglomerative",
        3,
        0.5658,
        0.0048,
        0.0002,
        -0.0001,
        "Κακά αποτελέσματα")

# Log HDBSCAN results (custom, no PCA)
add_results("20 Newsgroups",
        "Custom Fasttext",
        "None",
        "HDBSCAN",
        2,
        0.5751,
        0.0074,
        0.0004,
        -0.0001,
        "Κακά αποτελέσματα")

# Train PCA on the custom embeddings (20NG) to inspect cumulative variance
pca_full = PCA()
# Fit on df_vectors_newsg_trained
pca_full.fit(df_vectors_newsg_trained)

# Compute cumulative explained variance
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

# Plot cumulative variance
plt.figure(figsize=(8, 5))
# Curve
plt.plot(cumulative_variance, marker='o')
# X label
plt.xlabel("Number of Components")
# Y label
plt.ylabel("Cumulative Explained Variance")
# Title
plt.title("PCA: Cumulative Variance Explained (20NG, Custom FastText)")
# Grid
plt.grid(True)
# 90% reference line
plt.axhline(y=0.9, linestyle='--', label="90% Variance")
# Legend
plt.legend()
# Show
plt.show()

# Define PCA with 45 components (based on ~90% explained variance)
pca_45n = PCA(45)
# Transform using pca_45n
df_vectors_news20_trained_pca = pca_45n.fit_transform(df_vectors_newsg_trained)

# Set up 2D t-SNE on the PCA embeddings (45 components)
tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
# Compute 2D projections
df_vectors_newsg_trained_pca_tsne = tsne.fit_transform(df_vectors_news20_trained_pca)

# Scatter plot
plt.figure(figsize=(8, 6))
# Points
plt.scatter(df_vectors_newsg_trained_pca_tsne[:, 0], df_vectors_newsg_trained_pca_tsne[:, 1], s=10, alpha=0.6)
# Title
plt.title("t-SNE Custom FastText με PCA=45 σε 2D (20NG)")
# Grid
plt.grid(True)
# Show
plt.show()

# Evaluate KMeans (k=2..24) on PCA (45 components) embeddings
evaluate_kmeans_range(df_vectors_news20_trained_pca, k_range=range(2,25), init_method="k-means++",
                      n_init=10, max_iter=300, tol=1e-4, random_state=42)

# Run KMeans with n_clusters=3 on PCA (45 components)
kmeans_model_newsg_trained_pca = run_kmeans(df_vectors_news20_trained_pca, n_clusters=3)

# Plot dendrogram on PCA (45 components) embeddings
plot_dendrogram_only(df_vectors_news20_trained_pca, linkage_method='ward', metric='euclidean',
                     figsize=(12, 6), title="Dendrogram (20NG, Custom+PCA=45)", truncate_lastp=15)

# Evaluate Agglomerative (k=2..24) on PCA (45 components)
evaluate_agglomerative_range(df_vectors_news20_trained_pca, k_range=range(2, 25), linkage='ward',
                             metric='euclidean', show_scores=True)

# Run Agglomerative with n_clusters=3 on PCA (45 components)
agg_model_newsg_trained_pca = run_agglomerative_clustering(df_vectors_news20_trained_pca, n_clusters=3,
                                                           show_dendrogram=False)

# Run HDBSCAN on PCA (45 components)
hdbscan_model_newsg_trained_pca = run_hdbscan_clustering(df_vectors_news20_trained_pca, min_cluster_size=10,
                                                         min_samples=5, compute_score=True, show_probabilities=False)

# Compare KMeans/Agglo/HDBSCAN on PCA (45 components)
models = [kmeans_model_newsg_trained_pca, agg_model_newsg_trained_pca, hdbscan_model_newsg_trained_pca]
# Titles
titles = ["KMeans (k = 3)", "Agglomerative (k = 3)", "HDBSCAN (k = 2)"]
# Labels
true_labels_list = [x_test_news["label"], x_test_news["label"], x_test_news["label"]]
# Run evaluation
evaluate_clustering_models(true_labels_list, models, titles)

# Log KMeans results (custom, PCA=45)
add_results("20 Newsgroups",
        "Custom Fasttext",
        "PCA 45",
        "KMeans",
        3,
        0.0048,
        0.0002,
        0.0002,
        -0.0001,
        "Κακά αποτελέσματα")

# Log Agglomerative results (custom, PCA=45)
add_results("20 Newsgroups",
        "Custom Fasttext",
        "PCA 45",
        "Agglomerative",
        3,
        0.6217,
        0.0048,
        0.0002,
        -0.0001,
        "Κακά αποτελέσματα")

# Log HDBSCAN results (custom, PCA=45)
add_results("20 Newsgroups",
        "Custom Fasttext",
        "PCA 45",
        "HDBSCAN",
        2,
        0.6296,
        0.0075,
        0.0007,
        -0.0001,
        "Κακά αποτελέσματα")

# Set up a TF-IDF vectorizer with selected settings
tfidf_vectorizer = TfidfVectorizer(
    stop_words='english',  # remove English stopwords
    max_df=0.6,            # ignore very frequent terms
    min_df=15,             # ignore very rare terms
    max_features=3000,     # cap the feature space
    ngram_range=(1, 2)     # unigrams + bigrams
)

# Join tokens back into clean text
x_test_news["Cleaned_Text"] = x_test_news["Tokens"].apply(lambda tokens: " ".join(tokens))

# Compute the TF-IDF sparse matrix
tfidf_matrix = tfidf_vectorizer.fit_transform(x_test_news["Cleaned_Text"])

# Convert to a DataFrame for downstream steps
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Set up 2D t-SNE on the TF-IDF features
tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
# Compute 2D projections
tfidf_newsg_tsne = tsne.fit_transform(tfidf_df)

# Plot the t-SNE scatter
plt.figure(figsize=(8, 6))
# TF-IDF t-SNE points
plt.scatter(tfidf_newsg_tsne[:, 0], tfidf_newsg_tsne[:, 1], s=10, alpha=0.6)
# Title
plt.title("t-SNE TF-IDF χωρίς PCA (20NG) σε 2D")
# Grid
plt.grid(True)
# Show
plt.show()

# Evaluate KMeans for k=2..24 on TF-IDF
evaluate_kmeans_range(tfidf_df, k_range=range(2,25), init_method="k-means++",
                      n_init=10, max_iter=300, tol=1e-4, random_state=42)

# Run KMeans with n_clusters=13
kmeans_model_newsg_tfidf = run_kmeans(tfidf_df, n_clusters=13)

# Plot a dendrogram on TF-IDF
plot_dendrogram_only(tfidf_df, linkage_method='ward', metric='euclidean',
                     figsize=(12, 6), title="Dendrogram (20NG, TF-IDF)", truncate_lastp=15)

# Evaluate Agglomerative (k=2..24) on TF-IDF
evaluate_agglomerative_range(tfidf_df, k_range=range(2, 25), linkage='ward',
                             metric='euclidean', show_scores=True)

# Run Agglomerative with n_clusters=2 on TF-IDF
agg_model_newsg_tfidf = run_agglomerative_clustering(tfidf_df, n_clusters=2, show_dendrogram=False)

# Run HDBSCAN on TF-IDF (lower min_samples to encourage cluster formation)
hdbscan_model_newsg_tfidf = run_hdbscan_clustering(tfidf_df, min_cluster_size=7, min_samples=1,
                                                   compute_score=True, show_probabilities=False)

# Compare KMeans/Agglo/HDBSCAN on TF-IDF
models = [kmeans_model_newsg_tfidf, agg_model_newsg_tfidf, hdbscan_model_newsg_tfidf]
# Titles
titles = ["KMeans (k = 13)", "Agglomerative (k = 2)", "HDBSCAN (k = 4)"]
# Labels
true_labels_list = [x_test_news["label"], x_test_news["label"], x_test_news["label"]]
# Run evaluation
evaluate_clustering_models(true_labels_list, models, titles)

# Log KMeans results (TF-IDF, no PCA)
add_results("20 Newsgroups",
        "TF-IDF",
        "None",
        "KMeans",
        13,
        0.0112,
        0.2671,
        0.2483,
        0.0586,
        "Πολύ χαμηλά αποτελέσματα")

# Log Agglomerative results (TF-IDF, no PCA)
add_results("20 Newsgroups",
        "TF-IDF",
        "None",
        "Agglomerative",
        2,
        0.0103,
        0.0871,
        0.0845,
        0.0225,
        "Κακά αποτελέσματα")

# Log HDBSCAN results (TF-IDF, no PCA)
add_results("20 Newsgroups",
        "TF-IDF",
        "None",
        "HDBSCAN",
        4,
        0.4121,
        0.0195,
        0.0075,
        -0.0001,
        "Κακά αποτελέσματα")

# PCA analysis for TF-IDF (to inspect explained variance)
pca_full = PCA()
# Fit PCA on the TF-IDF features
pca_full.fit(tfidf_df)

# Compute cumulative explained variance for TF-IDF
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

# Figure
plt.figure(figsize=(8, 5))
# Curve
plt.plot(cumulative_variance, marker='o')
# Labels / Title
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA: Cumulative Variance Explained (TF-IDF)")
# Grid
plt.grid(True)
# 90% reference line
plt.axhline(y=0.90, linestyle='--', label="90% Variance")
# Legend
plt.legend()
# Show
plt.show()

# PCA with 20 components on TF-IDF
pca_2d = PCA(n_components=20)
# Project TF-IDF into 20 components
tfidf_pca_2d = pca_2d.fit_transform(tfidf_df)

# Set up t-SNE on TF-IDF reduced with PCA(20)
tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
# 2D projections
df_tfidf_pca_2d_newsg_pca_tsne = tsne.fit_transform(tfidf_pca_2d)

# Figure
plt.figure(figsize=(8, 6))
# Scatter
plt.scatter(df_tfidf_pca_2d_newsg_pca_tsne[:, 0], df_tfidf_pca_2d_newsg_pca_tsne[:, 1], s=10, alpha=0.6)
# Title
plt.title("t-SNE TF-IDF with PCA (2D)")
# Grid
plt.grid(True)
# Show
plt.show()

# PCA with 800 components for comparison (as discussed)
pca_2d = PCA(n_components=800)
# Project TF-IDF into 800 components
tfidf_pca_800 = pca_2d.fit_transform(tfidf_df)

# Set up t-SNE on TF-IDF reduced with PCA(800)
tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
# 2D projections
df_tfidf_800_newsg_pca_tsne = tsne.fit_transform(tfidf_pca_800)

# Figure
plt.figure(figsize=(8, 6))
# Scatter
plt.scatter(df_tfidf_800_newsg_pca_tsne[:, 0], df_tfidf_800_newsg_pca_tsne[:, 1], s=10, alpha=0.6)
# Title
plt.title("t-SNE TF-IDF with PCA n=800 (2D)")
# Grid
plt.grid(True)
# Show
plt.show()

# Evaluate KMeans over k=2..24 on TF-IDF PCA(20)
evaluate_kmeans_range(tfidf_pca_2d, k_range=range(2,25), init_method="k-means++", n_init=10, max_iter=300, tol=1e-4, random_state=42)

# KMeans (k=20) on TF-IDF PCA(20)
kmeans_model_newsg_tfidf_pca = run_kmeans(tfidf_pca_2d, n_clusters=20)

# Dendrogram on TF-IDF PCA(20)
plot_dendrogram_only(tfidf_pca_2d, linkage_method='ward', metric='euclidean', figsize=(12, 6), title="Dendrogram", truncate_lastp=15)

# Evaluate Agglomerative over k=2..24 on TF-IDF PCA(20)
evaluate_agglomerative_range(tfidf_pca_2d, k_range=range(2, 25), linkage='ward', metric='euclidean', show_scores=True)

# Agglomerative (k=2) on TF-IDF PCA(20)
agg_model_newsg_tfidf_pca = run_agglomerative_clustering(tfidf_pca_2d, n_clusters=2, show_dendrogram=False)

# HDBSCAN on TF-IDF PCA(20)
hdbscan_model_newsg_tfidf_pca = run_hdbscan_clustering(tfidf_pca_2d, min_cluster_size=7, min_samples=1, compute_score=True, show_probabilities=False)

# Compare models on TF-IDF PCA(20)
models = [kmeans_model_newsg_tfidf_pca, agg_model_newsg_tfidf_pca, hdbscan_model_newsg_tfidf_pca]
# Titles
titles = ["KMeans (k = 20)", "Agglomerative (k = 2)", "HDBSCAN (k = 15)"]
# True labels
true_labels_list = [x_test_news["label"], x_test_news["label"], x_test_news["label"]]
# Evaluation
evaluate_clustering_models(true_labels_list, models, titles)

# Log KMeans results (TF-IDF PCA=20)
add_results("20 Newsgroups",
        "TF-IDF",
        "PCA 20",
        "KMeans",
        20,
        0.1546,
        0.2658,
        0.2372,
        0.0638,
        "Πολύ χαμηλά αποτελέσματα")

# Log Agglomerative results (TF-IDF PCA=20)
add_results("20 Newsgroups",
        "TF-IDF",
        "PCA 20",
        "Agglomerative",
        2,
        0.1878,
        0.0620,
        0.0589,
        0.0040,
        "Κακά αποτελέσματα")

# Log HDBSCAN results (TF-IDF PCA=20)
add_results("20 Newsgroups",
        "TF-IDF",
        "PCA 20",
        "HDBSCAN",
        15,
        0.2936,
        0.0960,
        0.0574,
        0.0025,
        "Κακά αποτελέσματα")

# Display the consolidated results table
display(clustering_results_df)