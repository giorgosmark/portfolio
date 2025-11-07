# Airline Tweet Sentiment Classifier — Transformers (BERT focus)

## Brief Description
This project performs **sentiment classification of airline tweets** using **Hugging Face Transformers encoders**.
The **implemented code runs BERT** in two regimes:

- **Classifier-only** fine-tuning (encoder frozen)
- **Unfreeze last encoder layer + classifier**

In the notebook, **additional result snapshots** (images) from **RoBERTa** and **BERTweet** are included **for comparison only**.
As noted in the analysis, results for “**last and second-to-last layer**” unfreezing were very similar, so they are **not detailed further**.

The model predicts one of three sentiment categories:

- Negative
- Neutral
- Positive

Designed and tested on **Google Colab GPU**. Best experienced via Colab.

## Contents
- `airline_transformers.ipynb`
	Full workflow in Greek: setup, BERT runs, metrics, loss curves, confusion matrices, and **embedded images** with **RoBERTa/BERTweet** results for side-by-side comparison.

- `main.py`
	- Clean execution script for **BERT only**:
	- Data loading & 80/10/10 stratified split
	- Tokenization (BERT tokenizer, max_length=64, **dynamic padding** via collator)
	- Model setup (bert-base-uncased)
	- Scenario A: **classifier-only** fine-tuning
	- Scenario B: **unfreeze last encoder layer + classifier**
	- Evaluation on validation/test, printed metrics, plots

- `utils.py`
	Helper utilities:
		- Data I/O & inspection (`load_data`, `inspect_data`)
		- Tokenization helper (`tokenize_text`)
		- Plotting (`plot_confusion_matrices`, `plot_losses_to_pdf`)

Console prints remain **in Greek** as per the notebook’s original flow. Docstrings/comments are **in English**.
  
## Data
- File: `Tweets.csv`
- Upload via Colab when prompted (recommended), or place under datasets/ and adjust the path in main.py.

## Environment & Execution
**Option A — Run the wrapper notebook (recommended)**
	Open the notebook and run all cells:
	- Installs the required libraries in Colab (pip)
	- Executes !python main.py
	- Prompts you to upload Tweets.csv if needed

**Option B — Open main.py directly in Colab**
	Colab can open .py files interactively:
	- File → Open notebook → GitHub → select main.py
	- Run cell-by-cell.
		!pip and google.colab imports are supported **only inside Colab**.
		
**Google Drive behavior**
	- On Colab, `drive.mount('/content/drive')` asks the user to grant access to **their own Drive**.
	- Outputs/checkpoints are saved to the **user’s** Drive (not the author’s).
	- If Drive is not mounted, the script falls back to a local Colab path (e.g., /content/bert_tweets_project).

## Libraries
Typical Colab setup (installed in the wrapper notebook):
```
!pip -q install "torch>=2.2" "transformers>=4.44.0" "datasets>=2.20.0" "scikit-learn>=1.3" matplotlib pandas numpy
```

## Notes
The repo code is **BERT-focused, RoBERTa/BERTweet** are presented **as images** in the notebook for comparison.
Some utilities are shared across projects and adapted here for Transformer-based classification.
Prints are kept **in Greek** (as in the original notebook) for the flow; comments/docstrings in **English**.

## Colab-only Commands
The following lines in `main.py` are specific to Google Colab and will raise errors in local Python environments:
```python
!pip -q install --upgrade pip
!pip -q install "torch>=2.2" "transformers>=4.44.0" "datasets>=2.20.0" "scikit-learn>=1.3" matplotlib pandas numpy
from google.colab import drive, files
drive.mount('/content/drive')
```
These commands:
	- Work only inside Colab (not in standard Python interpreters)
	- Request access to the user’s own Google Drive
	- Are included for reproducibility within Colab GPU runtime

## Context
This project was developed as part of the AI Data Factory program at the
Athens University of Economics and Business (AUEB).