# Airline Tweet Sentiment Classifier — BiLSTM + Attention (Word2Vec)

## Brief Description
This project performs **sentiment classification of airline tweets** using a **Bidirectional LSTM with Attention** architecture and **pre-trained Word2Vec embeddings**.  
It demonstrates a complete **end-to-end NLP workflow**: from text cleaning and lemmatization, to sequence modeling, training, and evaluation — all implemented in Python using **PyTorch**, **spaCy**, and **scikit-learn**.

The model classifies each tweet into one of three sentiment categories:
- Negative  
- Neutral  
- Positive  


---

## Contents
- `airline_nlp_rnn.ipynb`: notebook presenting the full analysis workflow (in Greek) with explanations, plots, and results.

- `main.py`  
  The main execution script — performs the complete pipeline:
  - Dataset loading and inspection  
  - Text cleaning, tokenization, and lemmatization  
  - Vocabulary creation and sequence padding  
  - Word2Vec embedding layer construction  
  - BiLSTM + Attention model definition and training (with early stopping)  
  - Evaluation, loss plotting, and confusion matrix visualization  

- `utils.py`  
  Helper module with all reusable functions and classes for:
  - **Data I/O and inspection** (`load_data`, `inspect_data`)  
  - **Text preprocessing** (`clean_text`, `spacy_preprocess_batch`, emoji mapping)  
  - **Sequence utilities** (`tokens_to_padded_ids_simple`, `build_embedding_layer`)  
  - **Model definition** (`TwoLayerBiLSTMAttention`)  
  - **Training loop and early stopping** (`train_model`, `EarlyStopping`)  
  - **Evaluation and plotting** (`get_preds`, `plot_confusion_matrices`, `plot_losses_to_pdf`)  

---

## Data
- **File:** `Tweets.csv`  
- **Location:** Place the dataset inside a `datasets` directory or update the path in `main.py` accordingly.

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