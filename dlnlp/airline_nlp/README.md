# Airline Tweet Sentiment Classifier — TF-IDF & Word2Vec Comparison

## Brief Description
This project focuses on **sentiment classification of airline tweets** using **supervised neural models** and two different text-vectorization methods.  
- TF-IDF (n-gram based)
- Word2Vec (pre-trained embeddings)
The goal is to build and compare models that can automatically classify tweet sentiment (negative, neutral, positive).
The workflow demonstrates a full text preprocessing and model training pipeline in Python using PyTorch and scikit-learn.

The model classifies each tweet into one of three sentiment categories:
- Negative  
- Neutral  
- Positive

## Contents
- `airline_nlp.ipynb`: notebook presenting the full analysis workflow (in Greek) with explanations, plots, and results.
- `main.py`: main execution file — includes the complete workflow for all stages: loading, cleaning, preprocessing, vectorization, ANN training, and evaluation.
- `utils.py`: Helper module providing reusable functions for:
  - **data loading and inspection** (load_data, inspect_data)
  - **text cleaning and lemmatization** (numeric filtering, column renaming, conversions)
  - **exploratory n-gram analysis** (top_ngrams, plot_bar)
  - **text vectorization** (TF-IDF and Word2Vec)
  - **model definition (ANN), training with early stopping, and evaluation** (get_preds, plot_losses_to_pdf)
  
## Data
- File: `Tweets.csv`
- Place the dataset in: `<datasets>`, or adjust the path inside `main.py`.

## Pre-trained Model Path
The Word2Vec embeddings require a pre-trained model.
Update the following line inside main.py to match your local filesystem:
```
w2v_path = r"C:\\Users\\<username>\\Projects\\word2vec-google-news-300.model"
```
Ensure the .model or .bin.gz file exists locally before running the script.

## Libraries
Use the central requirements file:
```
pip install -r ../requirements/requirements.txt
```

The requirements.txt file contains dependencies for all projects in the repository, not just this one.
Imports in this project are intentionally comprehensive — they cover the full NLP, PyTorch, and scikit-learn toolchain used throughout the analysis.

## Execution
After installing the required libraries, run:
```
python main.py
```
Check that the dataset and Word2Vec paths inside main.py are correct before execution.

## Notes
The notebook (.ipynb) showcases the full workflow (in Greek) with results, visualizations, and commentary.
The main.py and utils.py files are the clean, modular version of the code — suitable for portfolio presentation or direct script execution.
Some helper functions may share names with other projects but are customized for NLP sentiment analysis here.

## Context
This project was developed as part of the AI Data Factory program at the
Athens University of Economics and Business (AUEB).