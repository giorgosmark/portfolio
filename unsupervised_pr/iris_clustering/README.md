# Clustering on the Iris Dataset — K-Means, Agglomerative & HDBSCAN

## Brief Description
This project focuses on a **comparison of clustering algorithms** on the **Iris dataset.**
The goal is to apply and compare the following methods:
- **K-Means Clustering**
- **Agglomerative (Hierarchical) Clustering**
- **HDBSCAN**

The analysis includes outlier detection, feature normalization, exploration of the optimal number of clusters,
and evaluation of results using NMI, AMI, and ARI metrics.

---

## Contents
- `iris_clustering.ipynb`: notebook presenting **the full analysis workflow** (in Greek) with explanations, plots, and results.  
- `main.py`: main execution file — includes the complete workflow for all analysis stages.  
- `utils.py`: helper functions for:
  - loading data (from file or sklearn)
  - data cleaning and outlier detection
  - computation of correlations and statistical features
  - application of various scalers (Standard, MinMax, etc.)
  - training and evaluation of K-Means, Agglomerative & HDBSCAN
  - 2D & 3D visualizations

---

## Data
- Dataset: Iris (built-in sklearn dataset)
No `.csv` file is required, as the data are loaded directly via:
```python
from sklearn.datasets import load_iris
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