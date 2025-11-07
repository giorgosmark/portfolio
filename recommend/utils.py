"""
    utils.py — Helper functions for the MovieLens recommender system project

    Purpose
        - Centralizes all reusable utility functions used throughout the recommendation pipeline.
        - Keeps main.py focused on the experimental workflow (data loading, training, evaluation).
        - Provides modular, dataset-agnostic building blocks for data preparation, inspection,
          matrix factorization, and embedding-based feature engineering.

    Contents (indicative)
        - Data I/O & Inspection:
            download_movielens, load_data, inspect_data
        - Mapping & Indexing:
            make_maps (user/movie ID mappings)
        - Matrix Factorization:
            train_mf_sgd, train_mf_sgd_2, rmse_known, rmse_known_2
            EarlyStopPQ (early stopping for MF)
        - Embedding Utilities:
            mean_l2, l2_norm_rows, mean_cosine_to_seeds
            compute_genre_based_embedding (build embeddings from genre prototypes)
        - String & Metadata Helpers:
            normalize_title, extract_year_from_title

    Usage
        from utils import (
            download_movielens, load_data, inspect_data, make_maps,
            train_mf_sgd, train_mf_sgd_2, rmse_known, rmse_known_2,
            mean_l2, l2_norm_rows, mean_cosine_to_seeds,
            compute_genre_based_embedding, normalize_title, extract_year_from_title,
            EarlyStopPQ
        )

    Dependencies
        - pandas, numpy, scipy, requests, urllib, os, zipfile, re
        - sentence-transformers (for text embeddings)
        - dotenv (for API keys, e.g., TMDB)
        - time (for rate limiting API requests)

    Notes
        - This module defines helper functions and classes only; no workflow is executed here.
        - All dataset paths, experiment logic, and printing are handled in main.py.
        - Functions are written to be reusable across recommendation experiments
          (content-based, collaborative, or hybrid).

    Maintenance
        - Add new data processing or model evaluation helpers here and import them explicitly in main.py.
        - Keep functions lightweight, well-documented, and independent of global state.
        - Prefer explicit return values over in-place modifications for clarity.
"""

# --- Core utilities & helpers used across the project ---
import urllib.request
import zipfile
import os
import pandas as pd
import numpy as np
import re


def download_movielens(url, target_dir, zip_name):
    """
    Download and unzip a MovieLens dataset (or any ZIP file).

    Parameters
    ----------
    url : str
        The URL from which the ZIP file will be downloaded.
    target_dir : str
        Destination folder for the ZIP and the extracted contents.
    zip_name : str
        The local name of the ZIP file to be saved.

    Returns
    -------
    str
        The path of the folder where the extracted files are located.
    """
    # Create destination folder if it does not exist
    os.makedirs(target_dir, exist_ok=True)

    # Full path for the ZIP file
    zip_path = os.path.join(target_dir, zip_name)

    # Download ZIP if it does not already exist
    if not os.path.exists(zip_path):
        print(f"Κατέβασμα από {url}...")
        urllib.request.urlretrieve(url, zip_path)
        print("ok")
    else:
        print("Το αρχείο zip υπάρχει ήδη")

    # Unzip into the destination folder
    with zipfile.ZipFile(zip_path, 'r') as z:
        print(f"Αποσυμπίεση στο {target_dir}...")
        z.extractall(target_dir)
        print("Ok!!")

    return target_dir


def load_data(source='file', filepath=None,  dataset_func=None, sheet_name=None):
    """
    Load data from a file (CSV/Excel/.dat) or from a built-in sklearn dataset.

    Parameters
    ----------
    source : str, optional
        'file' for local file (default) or 'sklearn' for sklearn dataset.
    filepath : str or None
        File path when source='file'. Supports .csv, .xls, .xlsx, .dat
    dataset_func : callable or None
        Function from sklearn.datasets (e.g., load_iris) when source='sklearn'
    sheet_name : str or None
        Excel sheet name (needed for .xls/.xlsx)

    Returns
    -------
    tuple
        (df, target) where:
        df : pd.DataFrame with data or None
        target : np.ndarray or None with target labels (for sklearn datasets)
    """
    if source == 'file':
        if not filepath:
            print("\nΠαρακαλώ δώσε filepath για CSV.")
            return None, None
        try:
            if filepath.endswith('.csv'):
                try:
                    df = pd.read_csv(filepath, encoding="utf-8")
                    print("\nDataset φορτώθηκε από CSV αρχείο:", df.shape)
                except UnicodeDecodeError:
                    df = pd.read_csv(filepath, encoding="utf-8-sig")
                    print("\nDataset φορτώθηκε από CSV αρχείο:", df.shape)
                return df, None

            elif filepath.endswith(('.xls', '.xlsx')):
                if sheet_name is None:
                    # show available sheets for selection
                    xls = pd.ExcelFile(filepath)
                    print("\nΔιαθέσιμα φύλλα εργασίας (sheets):", xls.sheet_names)
                    print("Χρησιμοποίησε το όρισμα sheet_name για να διαλέξεις φύλλο.")
                    return None, None
                else:
                    df = pd.read_excel(filepath, sheet_name=sheet_name)
                    print("\n Dataset φορτώθηκε από XLS/XLSX αρχείο:", df.shape)
                    return df, None

            elif filepath.endswith('.dat'):
                try:
                    encoding = input("Δώσε encoding (π.χ. utf-8 ή latin-1): ")
                    separator = input("Δώσε διαχωριστικό (π.χ. ',' ή '::' ή '\\t'): ")

                    # Defaults if user left empty
                    if encoding.strip() == "":
                        encoding = "utf-8"
                    if separator.strip() == "":
                        separator = ","

                    df = pd.read_csv(filepath, sep=separator, encoding=encoding, engine="python", header=None)
                    print("\nDataset φορτώθηκε από .dat αρχείο:", df.shape)
                    return df, None

                except UnicodeDecodeError:
                    print("\n Σφάλμα: Το encoding που δόθηκε δεν ταιριάζει με το αρχείο.")
                    return None, None

                except pd.errors.ParserError:
                    print("\n Σφάλμα: Το διαχωριστικό που δόθηκε μάλλον δεν ταιριάζει.")
                    return None, None

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


def inspect_data(df, target_column=None):
    """
    Quick inspection of a DataFrame.

    Shows:
    - Shape (rows, cols)
    - Dtypes and NaN counts
    - Head
    - Describe
    - If target_column provided: label distribution (counts and percentages)

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to inspect.
    target_column : str, optional
        Target column (e.g., for classification). If provided, prints its distribution.
    """
    print(f"\nΣχήμα DataFrame: {df.shape}")
    print("\nΠληροφορίες DataFrame:")
    df.info()
    print("\nΠρώτες 5 γραμμές:")
    print(df.head())
    print("\nΠεριγραφικά στατιστικά:")
    print(df.describe())
    print("\nΈλεγχος για NaN:")
    print(df.isna().sum())

    if target_column is not None and target_column in df.columns:
        print(f"\nΚατανομή Target labels ({target_column}):")
        print(df[target_column].value_counts())
        print("\nΚατανομή σε ποσοστά:")
        print(df[target_column].value_counts(normalize=True) * 100)


def make_maps(df, col):
    """
    Build mapping dicts for unique values of a column.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    col : str
        Column name to encode.

    Returns
    -------
    tuple
        val_map : dict
            Mapping value -> index.
        val_imap : dict
            Mapping index -> value.
    """
    # Find unique values
    unique_vals = df[col].unique()

    # Build value -> index
    val_map  = {val: idx for idx, val in enumerate(unique_vals)}

    # Build index -> value
    val_imap = {idx: val for idx, val in enumerate(unique_vals)}

    return val_map, val_imap


def train_mf_sgd(
    R_coo,                 # TRAIN sparse COO (users x items) with ratings
    n_users, n_items,      # number of unique users/items in TRAIN
    K=20,                  # latent dimension
    epochs=100,            # training epochs
    batch_size=100_000,    # ratings per epoch (mini-batch size here is per-epoch sample)
    lr=0.01,               # learning rate (η)
    reg=0.05,              # L2 regularization (λ)
    seed=42,               # seed for reproducibility
    verbose=True,          # print progress
    val_known=None,        # optional: validation data/structure for RMSE each epoch
    patience=5,            # early stopping patience
    min_delta=1e-4         # minimal val RMSE improvement to be counted as progress
):
    """
    Matrix Factorization with SGD (no biases),
    Returns factor matrices P (users x K) and Q (items x K).
    """
    stopper = EarlyStopPQ(patience=patience, min_delta=min_delta)
    rng = np.random.default_rng(seed)

    # --- Initialize factors with small random values around 0
    P = 0.1 * rng.standard_normal((n_users, K))
    Q = 0.1 * rng.standard_normal((n_items, K))

    # --- Extract non-zero positions from TRAIN
    rows  = R_coo.row          # user indices per train rating
    cols  = R_coo.col          # item indices per train rating
    data  = R_coo.data.astype(float, copy=False)  # actual ratings (float)

    n = len(data)              # number of train ratings
    if n == 0:
        if verbose:
            print("WARN: Empty training set. Returning initial P, Q.")
        return P, Q

    for epoch in range(1, epochs + 1):
        # 1) shuffle all indices
        idx_all = np.arange(n)
        rng.shuffle(idx_all)

        # 2) iterate mini-batches to cover whole train
        num_batches = int(np.ceil(n / batch_size))
        if verbose:
            print(f"Epoch {epoch}/{epochs} — {num_batches} batches των ~{batch_size:,} ratings")

        # iterate batches
        for b in range(num_batches):
            start = b * batch_size
            end   = min(start + batch_size, n)
            idx   = idx_all[start:end]

            u_batch = rows[idx]
            i_batch = cols[idx]
            r_batch = data[idx]

            take = end - start

            # --- core SGD: one update per (u,i,r)
            for k in range(take):
                u = u_batch[k]
                i = i_batch[k]
                r = r_batch[k]

                # prediction via dot product
                pred = float(np.dot(P[u], Q[i]))

                # error
                err = r - pred

                # keep old P[u] for the Q[i] update
                Pu_old = P[u].copy()

                # updates with L2 regularization
                P[u] += lr * (err * Q[i]   - reg * P[u])
                Q[i] += lr * (err * Pu_old - reg * Q[i])

        if val_known is not None:
            val_rmse = rmse_known(P, Q, val_known)
            if verbose:
                print(f"[val] epoch {epoch} RMSE={val_rmse:.4f}")

            stopper(val_rmse, P=P, Q=Q, epoch=epoch)
            if stopper.early_stop:
                if verbose:
                    print("Early stopping triggered.")
                break
        else:
            if verbose:
                print(f"Epoch {epoch}/{epochs} — processed {take:,} ratings")

    # --- after loop: restore best if available
    if (stopper.best_P is not None) and (stopper.best_Q is not None):
        P, Q, best_rmse, best_epoch = stopper.restore()
        print(f"Restored best weights from epoch {best_epoch} "
              f"(best val rmse: {best_rmse:.4f}).")

    return P, Q


def rmse_known(P, Q, df_known):
    """
    Compute RMSE only for rows where both UserIdx and MovieIdx exist (known-known).

    Parameters
    ----------
    P : np.ndarray
        User factors (n_users x K).
    Q : np.ndarray
        Item factors (n_items x K).
    df_known : pd.DataFrame
        DataFrame with columns UserIdx, MovieIdx, rating (no NaNs on indices).

    Returns
    -------
    float
        RMSE value over known triples (u,i,r).
    """
    # Convert DataFrame columns into numpy arrays
    u = df_known['UserIdx'].to_numpy(int)    # user indices
    i = df_known['MovieIdx'].to_numpy(int)   # item indices
    y = df_known['rating'].to_numpy(float)   # true ratings

    # Predictions via dot products
    yhat = (P[u] * Q[i]).sum(axis=1)

    # Mean Squared Error
    mse = ((y - yhat) ** 2).mean()

    # Root Mean Squared Error
    rmse = mse ** 0.5
    return rmse


def train_mf_sgd_2(
    R_coo,                 # TRAIN sparse COO (users x items) with ratings
    n_users, n_items,      # number of unique users/items in TRAIN
    K=20,                  # latent dimension
    epochs=100,            # max epochs
    batch_size=100_000,    # ratings per epoch
    lr=0.01,               # learning rate (η)
    reg=0.05,              # L2 regularization (λ)
    seed=42,               # seed for reproducibility
    verbose=True,          # print progress
    val_known=None,        # (optional) validation data/structure for RMSE per epoch
    patience=5,            # early stopping patience
    min_delta=1e-4,        # minimal improvement to count as progress
    E=None,                # content embeddings for items (n_items x d)
    reg_content=0.0,       # content regularization strength
    lr_w=None              # learning rate for W
):
    """
    Matrix Factorization with SGD (no biases).
    Returns factor matrices P (users x K), Q (items x K),
    and optionally the content mapping W (K x d) if content regularization is used.
    """
    # Early stopping mechanics:
    # - patience: allows up to 'patience' consecutive epochs without improvement > min_delta
    # - min_delta: minimal decrease in val RMSE to count as "improvement"
    stopper = EarlyStopPQ(patience=patience, min_delta=min_delta)

    # RNG
    rng = np.random.default_rng(seed)

    # --- If E is provided and content regularization is enabled
    if (E is not None) and (reg_content > 0):
        d = E.shape[1]   # embedding dimensionality (e.g., 384)
        # Initialize W with small random values
        W = 0.1 * rng.standard_normal((K, d)).astype(np.float32)
        # If lr_w not provided, reuse lr
        if lr_w is None:
            lr_w = lr
    else:
        W = None

    # --- Initialize P and Q with small random values
    P = (0.1 * rng.standard_normal((n_users, K))).astype(np.float32)
    Q = (0.1 * rng.standard_normal((n_items, K))).astype(np.float32)

    # --- Get indices and data from COO
    rows = R_coo.row
    cols = R_coo.col
    data = R_coo.data.astype(np.float32, copy=False)

    n = len(data)  # number of ratings
    if n == 0:
        if verbose:
            print("WARN: Empty training set. Returning initial P, Q.")
        return P, Q

    # --- Training loop
    for epoch in range(1, epochs + 1):
        # 1) shuffle all indices
        idx_all = np.arange(n)
        rng.shuffle(idx_all)

        # 2) split in mini-batches
        num_batches = int(np.ceil(n / batch_size))
        if verbose:
            print(f"Epoch {epoch}/{epochs} — {num_batches} batches των ~{batch_size:,} ratings")

        # iterate all mini-batches
        for b in range(num_batches):
            start = b * batch_size
            end   = min(start + batch_size, n)
            idx   = idx_all[start:end]

            # batch users, items, ratings
            u_batch = rows[idx]
            i_batch = cols[idx]
            r_batch = data[idx]

            take = end - start  # batch size

            # update for each (u,i,r) in batch
            for k in range(take):
                u = u_batch[k]  # user idx
                i = i_batch[k]  # item idx
                r = r_batch[k]  # true rating

                # prediction = dot(P[u], Q[i])
                pred = float(np.dot(P[u], Q[i]))

                # prediction error
                err = r - pred

                # copy P[u] before update (for Q[i] update)
                Pu_old = P[u].copy()

                # compute WEi if W is available
                WEi = None
                if (W is not None):
                    Ei  = E[i]      # embedding of item i (dim d)
                    WEi = W @ Ei    # projection into latent space (dim K)

                # update P[u] with error + L2 regularization
                P[u] += lr * (err * Q[i] - reg * P[u])

                # update Q[i]
                if WEi is None:
                    # classic update without content reg
                    Q[i] += lr * (err * Pu_old - reg * Q[i])
                else:
                    # content-regularized update towards WEi
                    Q[i] += lr * (err * Pu_old - reg * Q[i] - reg_content * (Q[i] - WEi))

                # update W (if content reg is used)
                if WEi is not None:
                    diff = (Q[i] - WEi)  # difference between Q[i] and WEi
                    # outer product diff·Ei to update W
                    W += ( (lr_w if lr_w is not None else lr) * reg_content ) * np.outer(diff, Ei)

        # --- Validation RMSE and early stopping (if validation set is provided)
        if val_known is not None:
            val_rmse = rmse_known_2(P, Q, val_known)
            if verbose:
                print(f"[val] epoch {epoch} RMSE={val_rmse:.4f}")

            stopper(val_rmse, P=P, Q=Q, epoch=epoch)
            if stopper.early_stop:
                if verbose:
                    print("Early stopping triggered.")
                break
        else:
            if verbose:
                print(f"Epoch {epoch}/{epochs} — processed {take:,} ratings")

    # --- After training: restore best P,Q if they were snapshotted
    if (stopper.best_P is not None) and (stopper.best_Q is not None):
        P, Q, best_rmse, best_epoch = stopper.restore()
        print(f"Restored best weights from epoch {best_epoch} "
              f"(best val rmse: {best_rmse:.4f}).")

    return P, Q, W


def rmse_known_2(P, Q, df_known):
    """
    Compute RMSE only for rows where both UserIdx and MovieIdx exist (known-known),
    using compact dtypes for efficiency.

    Parameters
    ----------
    P : np.ndarray
        User factors (n_users x K).
    Q : np.ndarray
        Item factors (n_items x K).
    df_known : pd.DataFrame
        DataFrame with columns UserIdx, MovieIdx, rating.

    Returns
    -------
    float
        RMSE value over known triples (u,i,r).
    """
    # Convert to numpy arrays with specific dtypes for efficiency
    u = df_known['UserIdx'].to_numpy(dtype=np.int32)    # user indices
    i = df_known['MovieIdx'].to_numpy(dtype=np.int32)   # item indices
    y = df_known['rating'].to_numpy(dtype=np.float32)   # true ratings

    # Predictions via dot products
    yhat = (P[u] * Q[i]).sum(axis=1)

    # Mean Squared Error
    mse = ((y - yhat) ** 2).mean()

    # Root Mean Squared Error
    rmse = mse ** 0.5
    return rmse


class EarlyStopPQ:
    def __init__(self, patience=5, min_delta=0.0):
        # Control parameters
        self.patience  = patience       # how many consecutive non-improving epochs allowed
        self.min_delta = min_delta      # minimal required RMSE decrease

        # State
        self.best_rmse = None           # best RMSE seen so far
        self.bad_count = 0              # non-improving epochs counter
        self.early_stop = False         # flag to stop
        self.best_epoch = None          # epoch of best RMSE

        # Snapshots for best weights (P,Q)
        self.best_P = None
        self.best_Q = None

    def __call__(self, val_rmse, P=None, Q=None, epoch=None):
        """
        Update early-stopping state with the current val RMSE and (optionally) store best P,Q.

        Parameters
        ----------
        val_rmse : float
            New RMSE from the validation set.
        P, Q : np.ndarray or None
            User and item factor matrices to snapshot if improved.
        epoch : int or None
            Current epoch number.

        Returns
        -------
        None
        """
        # If no "best" yet, or improvement > min_delta
        if (self.best_rmse is None) or (self.best_rmse - val_rmse > self.min_delta):
            # Improvement
            self.best_rmse = float(val_rmse)
            self.bad_count = 0
            self.early_stop = False

            # Save snapshots if provided
            if (P is not None) and (Q is not None):
                self.best_P = P.copy()
                self.best_Q = Q.copy()

            # Save epoch of improvement
            if epoch is not None:
                self.best_epoch = epoch
        else:
            # No improvement
            self.bad_count += 1
            # Trigger early stopping when patience is exceeded
            if self.bad_count >= self.patience:
                self.early_stop = True

    def restore(self):
        """
        Return the best stored weights (P, Q) and the best RMSE.

        Returns
        -------
        tuple
            (best_P, best_Q, best_rmse, best_epoch)
        """
        if (self.best_P is None) or (self.best_Q is None):
            raise RuntimeError("No snapshot saved — check if EarlyStopping was called with P,Q.")
        return self.best_P, self.best_Q, self.best_rmse, self.best_epoch


def mean_l2(vectors):
    """
    Compute the mean vector of a list of embeddings and return
    its L2-normalized version.

    Parameters
    ----------
    vectors : list of np.ndarray
        List of vectors (embeddings) with the same dimensionality.

    Returns
    -------
    np.ndarray
        L2-normalized mean vector with ||m||_2 = 1.
    """
    # Stack vectors into a 2D array (n_vectors x dim)
    mat = np.vstack(vectors)

    # Mean per dimension
    m = mat.mean(axis=0)

    # Normalize to unit L2 norm
    return m / np.linalg.norm(m)


# builds an embedding for a movie by averaging the genre prototype embeddings
def movie_proto_embedding(genres, protos):
    """
    Compute a movie embedding based on the prototype embeddings of its genres.

    Parameters
    ----------
    genres : list of str
        List of genres associated with the movie.

    protos : pandas.Series or pandas.DataFrame
        Mapping of genre names to their corresponding prototype embeddings.

    Returns
    -------
    np.ndarray
        Final L2-normalized movie embedding.
    """
    # Collect prototype embeddings for each movie genre
    # (only if the genre exists in the 'protos' index)
    vecs = [protos[g] for g in genres if g in protos.index]

    # Average and L2-normalize (unit vector)
    return mean_l2(vecs)


def normalize_title(s: str) -> str:
    """Simple title normalization for search (lowercasing, removing symbols)."""
    s = str(s).lower()                              # lowercase
    s = re.sub(r'[^a-z0-9 ]+', ' ', s)              # keep only letters/numbers/spaces
    s = re.sub(r'\s+', ' ', s).strip()              # compress multiple spaces
    return s


def extract_year_from_title(title: str):
    """Attempt to extract year from a title like 'Movie (2008)'. Returns None if missing."""
    m = re.search(r'\((\d{4})\)', str(title))
    return int(m.group(1)) if m else None


def l2_norm_rows(X, eps = 1e-8):
    """L2-normalize each row vector to have length ~1."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)         # length of each vector
    return X / np.clip(norms, eps, None)                     # safe division for tiny norms


def mean_cosine_to_seeds(items, seeds):
    """
    Compute the MEAN cosine similarity of each item to ALL seed vectors.
    - items: (N, D) matrix of item vectors to score
    - seeds: (S, D) matrix of seed vectors
    Returns: (N,) vector with the mean cosine similarity per item.
    """
    A = l2_norm_rows(items)           # normalize items
    B = l2_norm_rows(seeds)           # normalize seeds
    sims = A @ B.T                    # (N, S): all dot products
    return sims.mean(axis=1)          # (N,): mean per item