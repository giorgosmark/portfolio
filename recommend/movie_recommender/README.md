# Hybrid Movie Recommender — Matrix Factorization & Content-Based Embeddings (MovieLens 10M + TMDB)

## Brief Description
This project implements a **hybrid recommender system** that combines **collaborative filtering** (via Matrix Factorization) and **content-based embeddings** derived from movie overviews and genres.
It demonstrates a complete end-to-end workflow — from **data loading and analysis** to **model training, hybridization**, and **evaluation** on both known and cold-start movies.
The workflow is based on the **MovieLens 10M dataset**, enriched with **TMDB metadata** (movie descriptions) using the official TMDB API.  

----

## Contents
- `movie_recommender.ipynb`  
  Jupyter notebook presenting the **full analysis workflow** (in Greek) with explanations, results, and plots.

- `main.py` main execution script — includes the complete workflow for all stages (data handling, training, enrichment, and evaluation).

- `utils.py`  
  Helper utilities for:
  - Data loading and inspection (`download_movielens`, `load_data`, `inspect_data`)
  - Mapping user/movie IDs (`make_maps`)
  - Matrix factorization training and evaluation (`train_mf_sgd`, `rmse_known`, `EarlyStopPQ`)
  - Embedding and normalization helpers (`mean_l2`, `l2_norm_rows`, `compute_genre_based_embedding`)
  - Title parsing and genre extraction (`normalize_title`, `extract_year_from_title`)

----

## Data

### Datasets
- **MovieLens 10M** — ratings and metadata
	Downloaded automatically from:
	https://files.grouplens.org/datasets/movielens/ml-10m.zip
	
- **MovieLens latest — links to TMDB IDs
	Used to connect MovieLens movies with TMDB metadata.
	
Both are automatically downloaded via `download_movielens()` in `utils.py`.
If download fails or you prefer local files, adjust the paths in `main.py` accordingly.

### Local Paths
In the current version, dataset paths are **hardcoded for Windows** 
(e.g. `C:\Users\giorg\Desktop\dataset_ml`).
To run the project elsewhere:
- Update all paths in `main.py` to point to your local dataset directory.
- Use `os.path.join()` for better portability if needed.
	
----

## TMDB API Key
The project queries the **TMDB API** to retrieve movie overviews and enrich the dataset with textual descriptions.
You need a valid API key from:
https://www.themoviedb.org/settings/api

**Options:**
- **(Preferred)** Define an environment variable:
	- On Windows PowerShell:
		`$env:TMDB_API_KEY = "your_api_key"`
	- On Linux/macOS:
		`export TMDB_API_KEY="your_api_key"`

- Alternatively, store the key in a .env file in the project root as:
		`TMDB_API_KEY=your_api_key`
		(If you use this option, ensure your environment automatically loads .env files — otherwise you must call load_dotenv() manually.)

The script accesses the key with:
API_KEY = os.getenv("TMDB_API_KEY")

**"You may need to restart your system for the environment variable to take effect."**

----

## Libraries
Use the central requirements file:
```
pip install -r ../requirements/requirements.txt
```

The requirements.txt file contains dependencies for all projects in the repository, not just this one.
Also, the imports are intentionally extensive — not limited only to the functions used in this specific project.

----

## Execution
After installing the required libraries, run:
```
python main.py
```
### Notes
- Check that dataset paths and the TMDB API key are correctly set before running.
- The first run may take a long time due to TMDB API calls and embedding generation.
- Sentence embeddings are built using the model:
	`all-MiniLM-L6-v2`
	from the SentenceTransformers library.

----

## Notes

The notebook (.ipynb) presents the entire hybrid workflow (in Greek), including:
- Exploratory Data Analysis (EDA)
- Rating trends by genre and year
- Baseline MF training and evaluation
- TMDB enrichment and embedding generation
- Hybrid MF model with content regularization
- Cold-start recommendations and similarity-based retrieval

The `main.py` and `utils.py` files are the clean, modular version of the implementation — suitable for a portfolio or reproducible run in a fresh environment.

----

## Context
This project was developed as part of the AI Data Factory program at the
Athens University of Economics and Business (AUEB).

---

## Known Limitations & Roadmap

**Early stopping snapshot consistency (P/Q vs W).**  
In the current version, `EarlyStopPQ` snapshots and restores only the **P** (user factors) and **Q** (item factors) matrices. When training with content regularization (`train_mf_sgd_2` with `E`, `reg_content > 0`), the **W** mapping (from content space to latent space) is **not** restored to the same best epoch. This can leave **Q** and **W** out of sync after early stopping and may degrade test/OOV performance.

- **Planned fix:** extend `EarlyStopPQ` to also snapshot/restore **W**, and use it inside `train_mf_sgd_2` so that P, Q, **and** W are consistently restored to the best validation epoch.

**Function duplication.**  
There are parallel functions (`train_mf_sgd` vs `train_mf_sgd_2`, `rmse_known` vs `rmse_known_2`) that differ mostly in content-regularization support and dtype handling.

- **Planned refactor:** unify them into a single MF trainer with optional content arguments (`E`, `reg_content`, `lr_w`) and a single RMSE helper that accepts dtypes explicitly.
