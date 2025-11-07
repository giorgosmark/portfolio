# House Price Prediction — Supervised Regression Project

## Brief Description
This project focuses on **predicting house selling prices** using **supervised linear regression** models.  
It demonstrates a complete regression workflow — from **data cleaning and exploration** to **feature engineering**, **model training**, **evaluation**, and **cross-validation**.

The dataset (`housedata.xls`) includes various attributes such as house area, number of bedrooms, location, and property type, which are used to predict the selling price.

## Contents
- `house_price_prediction.ipynb`: notebook presenting **the full workflow** (in Greek), including analysis, explanations, and visualizations. 
- `main.py`: main execution file — includes the complete workflow for all analysis stages.
- `utils.py`: Helper module providing reusable functions for:
  - **Data loading and cleaning**
  - **Feature preprocessing** (numeric filtering, column renaming, conversions)
  - **Exploratory Data Analysis (EDA)** — boxplots, outlier detection, distributions, skew/kurtosis
  - **Correlation analysis** and **cross-validation metrics** (RMSE, MAE)
  
## Data
- File: `housedata.xls`
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