"""
    Main script for hybrid movie recommendation using collaborative filtering
    and content-based embeddings (MovieLens 10M + TMDB API).

    Purpose:
        - Executes the full end-to-end workflow for building and evaluating a recommender system
          that combines matrix factorization with text- and genre-based embeddings.
        - Demonstrates dataset preparation, EDA, matrix factorization (MF) with SGD and early stopping,
          enrichment with movie metadata and TMDB overviews, and hybrid content-regularized training.
        - Keeps reusable utilities (data loading, MF training, embedding functions, etc.)
          modularized in `utils.py` for clarity and reuse.

    Workflow Overview:
        1. Download and extract the MovieLens 10M dataset.
        2. Load ratings and movies, inspect data structure and quality.
        3. Analyze rating trends by genre and year (mean ratings and declines).
        4. Split data into train/validation/test sets based on timestamp years.
        5. Train a baseline matrix factorization model with SGD and evaluate RMSE.
        6. Load TMDB links to enrich movies with external IDs.
        7. Query TMDB API for movie overviews (descriptions) and build text embeddings using
           `SentenceTransformer` ("all-MiniLM-L6-v2").
        8. Compute genre prototype embeddings and infer embeddings for movies without text.
        9. Merge embeddings with ratings and retrain a hybrid MF model that integrates content
           regularization (aligning item factors with embedding space via matrix W).
        10. Evaluate the model on known and cold-start (OOV) movies.

    Notes:
        - Requires a valid TMDB API key stored in a `.env` file as TMDB_API_KEY.
        - All data-handling and reusable logic (e.g., load_data, train_mf_sgd, mean_l2)
          are defined in `utils.py` and imported here.
        - This script focuses on reproducibility, clarity, and modular structure suitable for
          portfolio demonstration.
        - Designed to run locally; API calls may fail without internet access.
"""

# --- Main workflow script for the MovieLens 10M project ---

# Filesystem utils (paths, env, etc.)
import os
# Data analysis
import pandas as pd
# Numeric computing
import numpy as np
# Sparse matrices
from scipy.sparse import coo_matrix
# Easier HTTP requests than urllib
import requests
# Load environment variables from .env
from dotenv import load_dotenv
# Timing (delays, timestamps)
import time
# Sentence embeddings (sentence transformers)
from sentence_transformers import SentenceTransformer


# Pandas display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)

# Import project utilities
from utils import (
    download_movielens, load_data, inspect_data, make_maps,
    train_mf_sgd, rmse_known, train_mf_sgd_2, rmse_known_2,
    EarlyStopPQ, mean_l2, movie_proto_embedding, normalize_title,
    extract_year_from_title, l2_norm_rows, mean_cosine_to_seeds
)

# -----------------------------
# Data loading & preprocessing
# -----------------------------

# --- Load MovieLens 10M dataset ---
url = "https://files.grouplens.org/datasets/movielens/ml-10m.zip"
dataset_dir = download_movielens(url, target_dir=r"C:\Users\giorg\Desktop\dataset_ml", zip_name="ml-10m.zip")

print(os.listdir(dataset_dir))

# --- Load ratings.dat ---
# NOTE: When prompted by load_data(), make sure to select:
#       - Encoding: utf-8
#       - Separator: ::
# These settings are required for proper parsing of the MovieLens .dat files.
# --- Load ratings.dat ---
ratings_df, _ = load_data(
    source='file',
    filepath=r"C:\Users\giorg\Desktop\dataset_ml\ml-10M100K\ratings.dat"
)

print("\n--- Πρώτες γραμμές από ratings ---")
print(ratings_df.head())

# Name columns
ratings_df.columns = ["userId", "movieId", "rating", "timestamp"]
print("\n--- Πρώτες γραμμές από το DataFrame ratings ---")
print(ratings_df.head())

# --- Load movies.dat ---
# NOTE: When prompted by load_data(), make sure to select:
#       - Encoding: utf-8
#       - Separator: ::
# These settings are required for proper parsing of the MovieLens .dat files.
# Load movies.dat
movies_df, _ = load_data(
    source='file',
    filepath=r"C:\Users\giorg\Desktop\dataset_ml\ml-10M100K\movies.dat"
)

print("\n--- Πρώτες γραμμές από movies ---")
print(movies_df.head())

movies_df.columns = ["movieId", "title", "genres"]
print("\n--- Πρώτες γραμμές από το DataFrame movies ---")
print(movies_df.head())

# Quick inspections
inspect_data(ratings_df)
inspect_data(movies_df)

# Duplicates checks
dup_ratings = ratings_df.duplicated(subset=["userId", "movieId"])
print("Διπλότυπα ratings:", dup_ratings.sum())  # Αποτέλεσμα: 0

dup_movies = movies_df.duplicated()
print("Διπλότυπα movies:", dup_movies.sum())  # Αποτέλεσμα: 0

# Create new "year" column from timestamp
ratings_df["year"] = pd.to_datetime(ratings_df["timestamp"], unit="s").dt.year
print(ratings_df.head())
print(f"Μικρότερο year: {ratings_df['year'].min()}")
print(f"Μεγαλύτερο year: {ratings_df['year'].max()}")

# --- Merge ratings + movies ---
df = ratings_df.merge(movies_df, on="movieId", how="left")
inspect_data(df)

# Drop timestamp (year already extracted)
df = df.drop(columns=["timestamp"])

# -----------------------------
# Genre analysis (explode, clean, means per year)
# -----------------------------
df_simple = df.copy()

# Create list of genres per row
df_simple["genres_list"] = df_simple["genres"].fillna("").apply(lambda x: x.split("|"))
print(df_simple.head())

# Explode genres
df_genres = df_simple.explode("genres_list")
print(df_genres.head())
print("Γραμμές πριν:", len(df), "— μετά το explode:", len(df_genres))
print("Μοναδικά genres:", sorted(df_genres["genres_list"].unique()))

# Remove unwanted pseudo-genres
mask = df_genres["genres_list"] == "(no genres listed)"
print("Πλήθος χωρίς είδος:", mask.sum())
print(df_genres[mask].head())

df_genres = df_genres[df_genres["genres_list"] != "(no genres listed)"]
df_genres = df_genres[df_genres["genres_list"] != "IMAX"]
print("Μοναδικά genres:", sorted(df_genres["genres_list"].unique()))

# Mean rating per genre-year
mean_per_year = df_genres.groupby(["genres_list","year"])["rating"].mean().reset_index()
print(mean_per_year.head())

# First/last year per genre
g = mean_per_year.copy()
first_year = g.groupby("genres_list")["year"].min().rename("first_year").reset_index()
last_year  = g.groupby("genres_list")["year"].max().rename("last_year").reset_index()
print(first_year.head())
print(last_year.head())

# Mean at first year per genre
g_first = g.merge(first_year, on=["genres_list"], how="inner")
g_first = g_first[g_first["year"] == g_first["first_year"]]
g_first = g_first[["genres_list", "first_year", "rating"]].rename(columns={"rating": "mean_first"})
print(g_first.head())

# Mean at last year per genre
g_last = g.merge(last_year, on=["genres_list"], how="inner")
g_last = g_last[g_last["year"] == g_last["last_year"]]
g_last = g_last[["genres_list","last_year","rating"]].rename(columns={"rating":"mean_last"})
print(g_last.head())

# Delta = mean_first - mean_last
results = g_first.merge(g_last, on="genres_list", how="inner").assign(delta=lambda x: x["mean_first"]-x["mean_last"])
print(results.head())

# Top-5 largest decreases
worst5 = results.sort_values("delta", ascending=False).head(5)
print(worst5)

# -----------------------------
# Adjust for rating counts (threshold by N)
# -----------------------------
# Count per genre-year
count_per_year = (
    df_genres.groupby(["genres_list","year"])
    .size()
    .reset_index(name="count")
)
print(count_per_year.sort_values("count").head(30))

# Threshold for reliability
N = 1500
valid_years = count_per_year[count_per_year["count"] >= N]
df_valid = df_genres.merge(valid_years[["genres_list", "year"]], on=["genres_list", "year"])
print(df_valid.head())

# Mean per (genre, year) after filtering
mean_per_year_2 = df_valid.groupby(["genres_list", "year"])["rating"].mean().reset_index()
g_2 = mean_per_year_2.copy()

first_year_2 = (
    g_2.groupby("genres_list")["year"]
    .min()
    .rename("first_year")
    .reset_index()
)
last_year_2 = (
    g_2.groupby("genres_list")["year"]
    .max()
    .rename("last_year")
    .reset_index()
)
print(first_year_2.head(30))

# Compute mean_first / mean_last over filtered data
g_first_2 = g_2.merge(first_year_2, on=["genres_list"], how="inner")
g_first_2 = g_first_2[g_first_2["year"] == g_first_2["first_year"]]
g_first_2 = g_first_2[["genres_list", "first_year", "rating"]].rename(columns={"rating": "mean_first"})

g_last_2 = g_2.merge(last_year_2, on=["genres_list"], how="inner")
g_last_2 = g_last_2[g_last_2["year"] == g_last_2["last_year"]]
g_last_2 = g_last_2[["genres_list", "last_year", "rating"]].rename(columns={"rating": "mean_last"})
print(g_last_2.head())

results_2 = g_first_2.merge(g_last_2, on="genres_list", how="inner").assign(delta=lambda x: x["mean_first"]-x["mean_last"])
print(results_2.head())

worst5_2 = results_2.sort_values("delta", ascending=False).head(5)
print(worst5_2)

# -----------------------------
# Train/Val/Test split for MF (year-based)
# -----------------------------
df_2 = df.copy()

# Split by year
train = df_2[df["year"] < 2008]
test = df_2[df["year"] >= 2008]

val_frac = 0.1
train_fit = train.sample(frac=1 - val_frac, random_state=42)
val_raw = train.drop(train_fit.index)

print(f"Train size: {len(train_fit):,} | years: {int(train_fit['year'].min())}–{int(train_fit['year'].max())}")
print(f"val size: {len(val_raw):,} | years: {int(val_raw['year'].min())}–{int(val_raw['year'].max())}")
print(f"Test  size: {len(test):,}  | years: {int(test['year'].min())}–{int(test['year'].max())}")

# Visual samples
print(train_fit.sample(3, random_state=0))
print(val_raw.sample(3, random_state=0))
print(test.sample(3, random_state=0))

# Build index maps
user_map,  user_imap  = make_maps(train_fit, "userId")
movie_map, movie_imap = make_maps(train_fit, "movieId")

train_fit = train_fit.copy()
train_fit['UserIdx']  = train_fit['userId'].map(user_map)
train_fit['MovieIdx'] = train_fit['movieId'].map(movie_map)
print(train_fit.head())

# Validation mapped
val = val_raw.copy()
val['UserIdx'] = val['userId'].map(user_map)
val['MovieIdx'] = val['movieId'].map(movie_map)
print(val.head())

# Split validation known/OOV
val_mask_known = val['UserIdx'].notna() & val['MovieIdx'].notna()
val_known = val[val_mask_known].copy()
val_oov = val[~val_mask_known].copy()

# Prepare test similarly
test = test.copy()
test['UserIdx'] = test['userId'].map(user_map)
test['MovieIdx'] = test['movieId'].map(movie_map)

mask_known = test['UserIdx'].notna() & test['MovieIdx'].notna()
test_known = test[mask_known].copy()
test_oov = test[~mask_known].copy()

n_users = len(user_map)
n_items = len(movie_map)
print(f"Users in train_fit (unique): {n_users:,}")
print(f"Movies in train_fit (unique): {n_items:,}")
print(f"Validation rows total:   {len(val):,}")
print(f"  Known IDs:             {len(val_known):,}")
print(f"  OOV (cold-start):      {len(val_oov):,}")
print(f"Test rows total:         {len(test):,}")
print(f"  Known IDs:             {len(test_known):,}")
print(f"  OOV (cold-start):      {len(test_oov):,}")

# Convert to numpy arrays
u = train_fit['UserIdx'].to_numpy(int)
i = train_fit['MovieIdx'].to_numpy(int)
r = train_fit['rating'].to_numpy(float)

# Build COO sparse matrix
R_coo = coo_matrix((r, (u, i)), shape=(n_users, n_items))

# Sparsity stats
num_nonzero = R_coo.nnz
total_cells = n_users * n_items
sparsity = 1 - (num_nonzero / total_cells)
print(f"Train ratings (non-zeros): {num_nonzero:,}")
print(f"Matrix shape: {n_users:,} users × {n_items:,} items")
print(f"Sparsity: {sparsity:.6f} (δηλ. {100*sparsity:.4f}% κελιά άδεια)")

# Train MF (SGD)
P, Q = train_mf_sgd(
    R_coo,
    n_users=n_users,
    n_items=n_items,
    K=1,
    epochs=2,
    batch_size=300_000,
    lr=0.03,
    reg=0.03,
    seed=42,
    verbose=True,
    val_known=val_known,
    patience=5,
    min_delta=8e-4
)

# Test-known evaluation
u_test = test_known['UserIdx'].to_numpy(int)
i_test = test_known['MovieIdx'].to_numpy(int)
y_test = test_known['rating'].to_numpy(float)
yhat = (P[u_test] * Q[i_test]).sum(axis=1)
mse_known  = ((y_test - yhat) ** 2).mean()
rmse_known_val = mse_known ** 0.5
print(f"Test_known: N={len(y_test):,} | MSE={mse_known:.4f} | RMSE={rmse_known_val:.4f}")

# -----------------------------
# TMDB enrichment (links + overviews -> embeddings)
# -----------------------------

# Download MovieLens-latest to get links.csv
url = "https://files.grouplens.org/datasets/movielens/ml-latest.zip"
dataset_dir = download_movielens(
    url,
    target_dir=r"C:\Users\giorg\Desktop\dataset_ml",
    zip_name="ml-latest.zip"
)
print(os.listdir(dataset_dir))

# Load links.csv
links_df, _ = load_data(
    source='file',
    filepath=r"C:\Users\giorg\Desktop\dataset_ml\ml-latest/links.csv"
)
print(links_df.head())

# Merge movies with links (imdbId, tmdbId)
movies_links = movies_df.merge(links_df, on='movieId', how='left', validate='one_to_one')
movies_links['imdbId'] = movies_links['imdbId'].astype('Int64')
movies_links['tmdbId'] = movies_links['tmdbId'].astype('Int64')
print(movies_links.head(1))

total = len(movies_links)
missing_imdb = movies_links['imdbId'].isna().sum()
missing_tmdb = movies_links['tmdbId'].isna().sum()
print(f"Σύνολο ταινιών: {total}")
print(f"Missing imdbId: {missing_imdb}  ({missing_imdb/total:.2%})")
print(f"Missing tmdbId: {missing_tmdb}  ({missing_tmdb/total:.2%})")

movies_links = movies_links.copy()
movies_links = movies_links[['movieId', 'title', 'genres', 'tmdbId']]
movies_links['has_tmdb'] = movies_links['tmdbId'].notna()
print(movies_links['has_tmdb'].value_counts())
print(movies_links.head())

# Test a TMDB call (requires TMDB_API_KEY in env)
API_KEY = os.getenv("TMDB_API_KEY")
tmdb_id = int(movies_links.loc[movies_links['has_tmdb'], 'tmdbId'].iloc[0])
url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
params = {"api_key": API_KEY, "language": "en"}
r = requests.get(url, params=params, timeout=10)
print("HTTP", r.status_code)
data = r.json() if r.ok else {}
print("TMDB title:", data.get("title"))
print("Overview:", (data.get("overview") or "")[:200], "…")

# Sample 3 tmdb requests
sample = movies_links.loc[movies_links['has_tmdb'], ['movieId','title','tmdbId']].sample(3, random_state=42)
rows = []
for _, row in sample.iterrows():
    tmdb_id = int(row.tmdbId)
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
    params = {"api_key": API_KEY, "language": "en"}
    resp = requests.get(url, params=params, timeout=10)
    d = resp.json() if resp.ok else {}
    rows.append({
        "movieId": row.movieId,
        "local_title": row.title,
        "tmdb_title": d.get("title"),
        "overview_200": (d.get("overview") or "")[:200]
    })
print(pd.DataFrame(rows))

# Keep movies with tmdbId
df_with_tmdb = movies_links[movies_links['has_tmdb']].copy()
print("Σύνολο ταινιών με tmdbId:", len(df_with_tmdb))

# Fetch first 10 overviews for sanity
results = []
for _, row in df_with_tmdb.head(10).iterrows():
    tmdb_id = int(row["tmdbId"])
    local_title = row["title"]
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
    params = {"api_key": API_KEY, "language": "en"}
    r = requests.get(url, params=params, timeout=10)
    data = r.json()
    results.append({
        "movieId": row['movieId'],
        "local_title": local_title,
        "tmdb_title": data.get("title"),
        "overview": (data.get("overview") or "")[:120] + "…"
    })
print(pd.DataFrame(results))

# Batch fetch overviews
batch_size = 500
df_with_tmdb = movies_links[movies_links['has_tmdb']].copy()
results = []
n = len(df_with_tmdb)
print("Σύνολο:", n)

for start in range(0, n, batch_size):
    end = min(start + batch_size, n)
    chunk = df_with_tmdb.iloc[start:end]
    print(f"Batch {start}-{end} / {n}")
    for _, row in chunk.iterrows():
        tmdb_id = int(row['tmdbId'])
        url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
        params = {"api_key": API_KEY, "language": "en"}
        try:
            r = requests.get(url, params=params, timeout=10)
            if r.status_code == 429:
                time.sleep(1.0)
                r = requests.get(url, params=params, timeout=10)
            if not r.ok:
                overview = None
            else:
                data = r.json()
                overview = (data.get("overview") or "").strip() or None
        except Exception:
            overview = None

        results.append({
            "movieId": row['movieId'],
            "title": row['title'],
            "overview": overview
        })
        time.sleep(0.01)

# Merge overviews back
overviews_df = pd.DataFrame(results)
overviews_slim = overviews_df[['movieId', 'overview']].copy()
movies_with_overview = movies_links.merge(
    overviews_slim, on='movieId', how='left'
)
have_text = movies_with_overview['overview'].notna().sum()
print(f"Περιγραφές λήφθηκαν για: {have_text} / {len(movies_with_overview)} "
      f"({have_text/len(movies_with_overview):.2%})")

# Build sentence embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

# Split movies with/without text
mw = movies_with_overview.copy()
has_text_mask = mw['overview'].notna() & (mw['overview'].str.strip() != "")
df_text = mw.loc[has_text_mask, ['movieId','title','genres','overview']].copy()
df_no_text = mw.loc[~has_text_mask, ['movieId','title','genres']].copy()
texts = df_text['overview'].tolist()

# Encode texts
emb = model.encode(
    texts,
    batch_size=512,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)
df_text['embedding'] = list(emb)
print(df_text.head(1))

# Clean genre lists for text-available movies
df_text['genre_list'] = df_text['genres'].fillna("").str.split('|')
df_text["genre_list"] = df_text["genre_list"].apply(
    lambda lst: [
        g.strip()
        for g in lst
        if g and g.strip().upper() not in ("IMAX", "(NO GENRES LISTED)")
    ]
)

# Build genre prototypes
df_text['has_genres'] = df_text['genre_list'].apply(lambda lst: len(lst) > 0)
df_g = df_text.loc[df_text['has_genres'], ['genre_list', 'embedding']].explode('genre_list', ignore_index=True)
df_g = df_g.rename(columns={'genre_list': 'genre'})
genre_proto = df_g.groupby('genre')['embedding'].apply(mean_l2)
print(genre_proto)

# Handle movies without overview (use genre prototypes)
df_no_text["genre_list"] = df_no_text["genres"].fillna("").str.split("|")
df_no_text["genre_list"] = df_no_text["genre_list"].apply(
    lambda lst: [
        g.strip()
        for g in lst
        if g and g.strip().upper() not in ("IMAX", "(NO GENRES LISTED)")
    ]
)
print(df_no_text.head(2))

df_no_text["embedding"] = df_no_text["genre_list"].apply(lambda genres: movie_proto_embedding(genres, genre_proto))
print(df_no_text.head())

# Merge all embeddings
emb_no_text_df = df_no_text[["movieId", "title", "genre_list", "embedding"]].copy()
emb_text_df = df_text[["movieId", "title", "genre_list", "embedding"]].copy()
all_emb = pd.concat([emb_text_df, emb_no_text_df], ignore_index=True)
print(all_emb.head())

# Compact dtypes for big training
df["userId"]  = df["userId"].astype("int32")
df["rating"]  = df["rating"].astype("float32")
df["year"]    = df["year"].astype("int32")
df["movieId"] = df["movieId"].astype("int32")

# Enrich ratings with embeddings
emb_subset = all_emb[["movieId", "embedding"]].copy()
df3 = df.copy()
df3 = df3.merge(emb_subset, on="movieId", how="left")
print(df3.head())

# Repeat split & mapping with embeddings
df_3 = df3.copy()
train = df_3[df_3["year"] < 2008]
test  = df_3[df_3["year"] >= 2008]

val_frac   = 0.1
train_fit  = train.sample(frac=1 - val_frac, random_state=42)
val_raw    = train.drop(train_fit.index)

user_map,  user_imap  = make_maps(train_fit, "userId")
movie_map, movie_imap = make_maps(train_fit, "movieId")

train_fit = train_fit.copy()
train_fit['UserIdx']  = train_fit['userId'].map(user_map)
train_fit['MovieIdx'] = train_fit['movieId'].map(movie_map)

val = val_raw.copy()
val['UserIdx']  = val['userId'].map(user_map)
val['MovieIdx'] = val['movieId'].map(movie_map)

val_mask_known = val['UserIdx'].notna() & val['MovieIdx'].notna()
val_known = val[val_mask_known].copy()
val_oov   = val[~val_mask_known].copy()

test = test.copy()
test['UserIdx']  = test['userId'].map(user_map)
test['MovieIdx'] = test['movieId'].map(movie_map)

mask_known = test['UserIdx'].notna() & test['MovieIdx'].notna()
test_known = test[mask_known].copy()
test_oov   = test[~mask_known].copy()

n_users = len(user_map)
n_items = len(movie_map)

u = train_fit["UserIdx"].to_numpy(dtype=np.int32)
i = train_fit["MovieIdx"].to_numpy(dtype=np.int32)
r = train_fit["rating"].to_numpy(dtype=np.float32)

R_coo = coo_matrix((r, (u, i)), shape=(n_users, n_items))

# Build E matrix of item embeddings
d = len(train_fit["embedding"].iloc[0])
E = np.zeros((n_items, d), dtype=np.float32)
for midx, emb in zip(train_fit["MovieIdx"], train_fit["embedding"]):
    E[midx] = np.array(emb, dtype=np.float32)

# Train content-regularized MF
P, Q, W = train_mf_sgd_2(
    R_coo,
    n_users=n_users,
    n_items=n_items,
    K=40,
    epochs=2,
    batch_size=300_000,
    lr=0.03,
    reg=0.08,
    seed=42,
    verbose=True,
    val_known=val_known,
    patience=5,
    min_delta=8e-4,
    E=E,
    reg_content=1e-2,
    lr_w=0.03
)

# Test-known evaluation
u_test = test_known['UserIdx'].to_numpy(dtype=np.int32)
i_test = test_known['MovieIdx'].to_numpy(dtype=np.int32)
y_test = test_known['rating'].to_numpy(dtype=np.float32)
yhat = (P[u_test] * Q[i_test]).sum(axis=1)
mse_known  = ((y_test - yhat) ** 2).mean()
rmse_known_val = mse_known ** 0.5
print(f"Test_known: N={len(y_test):,} | MSE={mse_known:.4f} | RMSE={rmse_known_val:.4f}")

# Validation item-OOV evaluation
val_item_oov = val_oov[val_oov["UserIdx"].notna() & val_oov["MovieIdx"].isna()].copy()
val_item_oov = val_item_oov[val_item_oov["embedding"].notna()].copy()

if val_item_oov.empty:
    print("Val_item_OOV: 0 γραμμές με διαθέσιμο embedding.")
else:
    u_arr = val_item_oov["UserIdx"].to_numpy(dtype=np.int32)
    y_arr = val_item_oov["rating"].to_numpy(dtype=np.float32)
    E_oov = np.stack(val_item_oov["embedding"].to_numpy()).astype(np.float32)
    q_tilde = (W @ E_oov.T).T
    yhat = (P[u_arr] * q_tilde).sum(axis=1)
    mse  = ((y_arr - yhat) ** 2).mean()
    rmse = mse ** 0.5
    print(f"Val_item_OOV: N={len(y_arr):,} | MSE={mse:.4f} | RMSE={rmse:.4f}")

# Test item-OOV evaluation
test_item_oov = test_oov[test_oov["UserIdx"].notna() & test_oov["MovieIdx"].isna()].copy()
test_item_oov = test_item_oov[test_item_oov["embedding"].notna()].copy()

if test_item_oov.empty:
    print("Test_item_OOV: 0 γραμμές με διαθέσιμο embedding.")
else:
    u_arr = test_item_oov["UserIdx"].to_numpy(dtype=np.int32)
    y_arr = test_item_oov["rating"].to_numpy(dtype=np.float32)
    E_oov = np.stack(test_item_oov["embedding"].to_numpy()).astype(np.float32)
    q_tilde = (W @ E_oov.T).T
    yhat = (P[u_arr] * q_tilde).sum(axis=1)
    mse  = ((y_arr - yhat) ** 2).mean()
    rmse = mse ** 0.5
    print(f"Test_item_OOV: N={len(y_arr):,} | MSE={mse:.4f} | RMSE={rmse:.4f}")

# Build pool with embeddings + metadata
pool = all_emb.merge(
    movies_with_overview[["movieId", "genres", "overview"]],
    on="movieId", how="left"
).copy()

# Add MovieIdx if present in train
pool["MovieIdx"] = pool["movieId"].map(movie_map)

# Extract year for display
pool["year"] = pool["title"].apply(extract_year_from_title)

# Normalized titles for seed matching
pool["title_norm"] = pool["title"].apply(normalize_title)

# Find the 3 seed movies
seed_titles = ["Iron Man", "300", "Transformers"]
seed_norms = [normalize_title(t) for t in seed_titles]

seed_rows = []
for tnorm in seed_norms:
    cands = pool[pool["title_norm"] == tnorm]
    if cands.empty:
        cands = pool[pool["title_norm"].str.contains(tnorm)]
    if cands.empty:
        continue
    cands = cands.sort_values(by="year", ascending=False, na_position="last")
    best = cands.iloc[0]
    seed_rows.append(best[["movieId", "title", "MovieIdx", "embedding"]])

seed_df = pd.DataFrame(seed_rows).drop_duplicates(subset="movieId")
print(seed_df)

# Build seed vectors
Q_seed_list = []
for _, row in seed_df.iterrows():
    midx = row["MovieIdx"]
    emb  = np.asarray(row["embedding"], dtype=np.float32)
    if pd.notna(midx):
        q_vec = Q[int(midx)]
    else:
        q_vec = W @ emb
    Q_seed_list.append(q_vec.astype(np.float32))
Q_seeds = np.vstack(Q_seed_list)
Emb_seeds = np.vstack(seed_df["embedding"].to_numpy()).astype(np.float32)
seed_movie_ids = set(seed_df["movieId"].tolist())

# Method 1 — Hybrid (MF + Content)
item_factors = []
for _, row in pool.iterrows():
    midx = row["MovieIdx"]
    if pd.notna(midx):
        v = Q[int(midx)]
    else:
        v = W @ np.asarray(row["embedding"], dtype=np.float32)
    item_factors.append(v.astype(np.float32))
item_factors = np.vstack(item_factors)

hybrid_scores_all = mean_cosine_to_seeds(item_factors, Q_seeds)
hybrid_rank = pool.loc[~pool["movieId"].isin(seed_movie_ids)].copy()
hybrid_rank["score_hybrid"] = hybrid_scores_all[~pool["movieId"].isin(seed_movie_ids)]
hybrid_top10 = (
    hybrid_rank
    .sort_values("score_hybrid", ascending=False)
    .head(10)[["title", "year", "genres", "overview", "score_hybrid"]]
)
print(hybrid_top10)

# Method 2 — Content-only
emb_items = np.vstack(pool["embedding"].to_numpy()).astype(np.float32)
content_scores_all = mean_cosine_to_seeds(emb_items, Emb_seeds)
content_rank = pool.loc[~pool["movieId"].isin(seed_movie_ids)].copy()
content_rank["score_content"] = content_scores_all[~pool["movieId"].isin(seed_movie_ids)]
content_top10 = (
    content_rank
    .sort_values("score_content", ascending=False)
    .head(10)[["title", "year", "genres", "overview", "score_content"]]
)
print(content_top10)

# Method 3 — Item–Item (Q similarity)
pool_known = pool[pool["MovieIdx"].notna()].copy()
Q_items = np.vstack([Q[int(m)] for m in pool_known["MovieIdx"].to_numpy(dtype=np.int64)]).astype(np.float32)
qsim_scores_all = mean_cosine_to_seeds(Q_items, Q_seeds)
qsim_rank = pool_known.loc[~pool_known["movieId"].isin(seed_movie_ids)].copy()
qsim_rank["score_qsim"] = qsim_scores_all[~pool_known["movieId"].isin(seed_movie_ids)]
qsim_top10 = (
    qsim_rank
    .sort_values("score_qsim", ascending=False)
    .head(10)[["title", "year", "genres", "overview", "score_qsim"]]
)
print(qsim_top10)