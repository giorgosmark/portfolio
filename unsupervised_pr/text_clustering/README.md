# Clustering on BBC News & 20 Newsgroups — FastText, TF-IDF, PCA, K-Means, Agglomerative & HDBSCAN

## Brief Description
This project focuses on comparing clustering algorithms applied to two text datasets:
the BBC News dataset and the 20 Newsgroups dataset.
The goal is to apply and evaluate the following methods:
- **K-Means Clustering**
- **Agglomerative (Hierarchical) Clustering**
- **HDBSCAN**

The analysis includes different forms of text representation (FastText, TF-IDF),
dimensionality reduction using PCA, exploration of the optimal number of clusters,
and evaluation of the results using the NMI, AMI, and ARI metrics.

The workflow also involves outlier detection, feature normalization, optimal cluster estimation,
and performance evaluation through the NMI, AMI, and ARI indices.

---

## Contents
- `text_clustering.ipynb`: a notebook presenting **the full analysis pipeline** (in Greek) with explanations, plots, and results.
- `main.py`: the main execution file — contains the full workflow of the analysis.
- `utils.py`: helper functions for:
	-loading data (from file or sklearn)
	-text cleaning and tokenization
	-converting text into vectors (FastText or TF-IDF)
	-applying PCA and t-SNE
	-training and evaluating K-Means, Agglomerative & HDBSCAN
	-visualizations and result comparisons

---

## Data
Dataset 1: BBC News
- File: bbc_news_test.csv
- Place the dataset in: <datasets>, or adjust the path in main.py.
Dataset 2: 20 Newsgroups
- Loaded directly from sklearn:
```python
from sklearn.datasets import fetch_20newsgroups
```

## Libraries
Use the central requirements file:
```
pip install -r ../requirements/requirements.txt
```

The requirements.txt file contains dependencies for all projects in the repository, not just this one.
Also, the imports are intentionally extensive — not limited only to the functions used in this specific project.

## Execution
After installing the required libraries, run:
```
python main.py
```
Check that the dataset path inside main.py is correct before execution.

## Notes
The notebook (.ipynb) showcases the full workflow (in Greek) with results, plots, and commentary.
The main.py and utils.py files are the “clean” version of the code — suitable for a portfolio or fresh environment execution.
Some functions may share names with other projects but may have been modified or extended for this analysis.

## Context
This project was developed as part of the AI Data Factory program at the
Athens University of Economics and Business (AUEB).