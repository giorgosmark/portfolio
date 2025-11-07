# Customer Segmentation — Clustering with K-Means

## Brief Description: 
This project focuses on **customer segmentation** using the **K-Means Clustering** algorithm.
It uses the dataset `mall_customers.csv`, which contains customer information such as age, income, and spending score.
The goal is to identify distinct groups of customers with similar purchasing behavior.

## Contents
- `mall_customers.ipynb`: notebook presenting **the full workflow** (in Greek), including analysis, explanations, and visualizations. 
- `main.py`: main execution file — loads the data and performs all analysis stages (EDA, scaling, clustering, visualization).  
- `utils.py`: helper functions for:
  - data cleaning and outlier detection
  - correlation computation
  - application of various scalers 
  - training and evaluation of K-Means
  - 2D & 3D visualizations

## Data
- File: `mall_customers.csv`
- Place the dataset in: `<datasets>`, or adjust the path inside `main.py`.

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