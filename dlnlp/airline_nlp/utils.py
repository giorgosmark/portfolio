"""
    utils.py — Helper functions for the text classification (NLP) project

    Purpose
        - Centralizes ALL reusable functions related to text preprocessing, vectorization,
          model training, and evaluation for sentiment analysis (or similar NLP tasks).
        - Keeps main.py clean by separating the execution workflow from reusable logic.

    Contents (indicative)
        - Data handling:
            load_data, inspect_data
        - Text preprocessing:
            clean_text, spacy_preprocess_batch
        - Exploratory analysis:
            top_ngrams, plot_bar
        - Feature extraction:
            vectorize_texts (TF-IDF or Word2Vec)
        - Modeling:
            ANN (feedforward neural network)
            train_model (with early stopping and LR scheduler)
            EarlyStopping (class)
        - Evaluation & visualization:
            get_preds, plot_losses_to_pdf

    Usage
          from utils import (
              load_data, inspect_data, clean_text, spacy_preprocess_batch,
              top_ngrams, plot_bar, vectorize_texts,
              ANN, train_model, get_preds, plot_losses_to_pdf
          )

    Dependencies
        - Core: numpy, pandas, matplotlib, re, html, copy
        - NLP: spaCy, gensim (KeyedVectors)
        - ML: torch (PyTorch), scikit-learn
        - Visualization: matplotlib.backends.backend_pdf.PdfPages

    Notes
        - This module DEFINES functions and classes only — no workflow is executed here.
        - Dataset paths, training parameters, and workflow logic should live in main.py.
        - All functions are designed to be modular and dataset-agnostic.
        - Each function returns explicit outputs (no implicit global state).

    Maintenance
        - When adding new preprocessing, vectorization, or model-evaluation routines,
          define them here and import explicitly from main.py.
        - Keep implementations efficient, well-documented, and reusable across NLP projects.
"""

# --- Core libraries ---
import numpy as np
import pandas as pd
import re
import copy
import html
import matplotlib.pyplot as plt

# --- PyTorch ---
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

# --- Scikit-learn ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# --- NLP ---
import spacy
from gensim.models import KeyedVectors

# --- PDF export for plots ---
from matplotlib.backends.backend_pdf import PdfPages


# Load the English spaCy model once (keeps tagger + lemmatizer)
nlp = spacy.load("en_core_web_sm")


def load_data(source='file', filepath=None, dataset_func=None, sheet_name=None):
    """
    Load data from file (CSV/Excel) or from a built-in sklearn dataset.

    Parameters
    ----------
    source : str, optional
        'file' for local file (default) or 'sklearn' for sklearn dataset.
    filepath : str or None
        File path when source='file'. Supports .csv, .xls, .xlsx
    dataset_func : callable or None
        Function from sklearn.datasets (e.g., load_iris) when source='sklearn'
    sheet_name : str or None
        Excel sheet name (only for .xls/.xlsx)

    Returns
    -------
    tuple
        (df, target) where:
        df : pd.DataFrame with data
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
                    xls = pd.ExcelFile(filepath)
                    print("\nΔιαθέσιμα φύλλα εργασίας (sheets):", xls.sheet_names)
                    print("Χρησιμοποίησε το όρισμα sheet_name για να διαλέξεις φύλλο.")
                    return None, None
                else:
                    df = pd.read_excel(filepath, sheet_name=sheet_name)
                    print("\n Dataset φορτώθηκε από XLS/XLSX αρχείο:", df.shape)
                    return df, None
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
    - Shape (rows, columns)
    - Dtypes and number of NaNs
    - First rows (head)
    - Descriptive statistics (describe)
    - If target_column provided: label distribution (counts and percentages)

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to inspect.
    target_column : str, optional
        Target column (for classification). If present, show distribution.
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


def clean_text(text,
               clean_html=True,
               remove_amp=True,
               remove_mentions=True,
               remove_urls=True,
               strip_hashtags=True,
               remove_numbers=True,
               normalize_hyphens=True,
               remove_noise=True,
               normalize_intensity=True,
               to_lower=True,
               normalize_space=True):
    """
    Text cleaning/normalization with modular flags.

    Depending on flags, performs:
    - Unescape HTML entities and remove HTML tags (without deleting <3)
    - Replace curly apostrophe ’ with '
    - Remove & (replace with space)
    - Remove @mentions
    - Remove URLs (http/https/www)
    - Strip '#' from hashtags (keep the word), remove ASCII emoticons (standalone), remove currency symbols
    - Replace numbers with 'NUM' and compress consecutive NUMs to a single 'NUM'
    - Replace '_' with space
    - Remove artifacts like {…}, […], <…> and special symbols (keep emojis/basic punctuation)
    - Normalize intensity: compress mixed ?!/!? to '!?', reduce multiple ! and ? to one, squash 3+ identical letters to 2
    - Fix known artifacts ('flightled'/'flightle'/'flighted'/'flightr' -> 'flight', 'rebooke' -> 'rebook')
    - Slang replacement (thx/thanks→thank, pls/plz→please, u→you, ur→your, ppl→people)
    - Optional lowercase
    - Final compression of consecutive 'num' → single 'num' and space normalization

    Parameters
    ----------
    text : str
        The raw input text to clean.
    clean_html, remove_amp, remove_mentions, remove_urls, strip_hashtags,
    remove_numbers, normalize_hyphens, remove_noise, normalize_intensity,
    to_lower, normalize_space : bool, optional
        Toggle the respective steps (default: True)

    Returns
    -------
    str
        The cleaned/normalized text.
    """
    s = text

    if clean_html:
        s = html.unescape(s)
        s = re.sub(r"</?[A-Za-z!][^>]*>", "", s)

    s = s.replace("’", "'")

    if remove_amp:
        s = s.replace("&", " ")

    if remove_mentions:
        s = re.sub(r"(?<!\w)@\w+", "", s)

    if remove_urls:
        s = re.sub(r"(https?://\S+|www\.\S+)", "", s)

    if strip_hashtags:
        s = re.sub(r"#(\w+)", r"\1", s)
        s = re.sub(r'(?i)(?<!\w)(?:[:;=x][\-^]?[)D\(P/\\]|<3)(?!\w)', ' ', s)
        s = re.sub(r'[$€£¥₩₽₹¢฿]', ' ', s)

    if remove_numbers:
        s = re.sub(r"\d+", "NUM", s)
        s = re.sub(r"\bNUM(?:\W+NUM)+\b", "NUM", s)

    if normalize_hyphens:
        s = s.replace("_", " ")

    if remove_noise:
        s = re.sub(r"\{.*?\}|\[.*?\]|<.*?>", " ", s)
        s = re.sub(r"[©®™★☆•…–—·※“”\"]", " ", s)

    if normalize_intensity:
        s = re.sub(r'(?=[!?]*!)(?=[!?]*\?)[!?]{2,}', '!?', s)
        s = re.sub(r'!{2,}', '!', s)
        s = re.sub(r'\?{2,}', '?', s)
        s = re.sub(r'(.)\1{2,}', r'\1\1', s)

    s = re.sub(r"\bflight(?:led|le|ed|r)\b", "flight", s, flags=re.IGNORECASE)
    s = re.sub(r"\brebooke\b", "rebook", s, flags=re.IGNORECASE)

    slang_map = {
        "thx": "thank",
        "thanks": "thank",
        "pls": "please",
        "plz": "please",
        "u": "you",
        "ur": "your",
        "ppl": "people"
    }
    for k, v in slang_map.items():
        s = re.sub(rf"\b{k}\b", v, s, flags=re.IGNORECASE)

    if to_lower:
        s = s.lower()

    s = re.sub(r"\bnum(?:\W+num)+\b", "num", s)

    if normalize_space:
        s = re.sub(r"\s+", " ", s).strip()

    return s


def spacy_preprocess_batch(texts, batch_size=800):
    """
    Preprocess a batch of texts with spaCy (batched for speed).

    Steps:
    - Tokenization and lemmatization via nlp.pipe
    - Remove stopwords (but keep negations: not, n't, no, never)
    - Remove punctuation, except for ! and ?
    - Use lemma instead of raw token
    - Normalize consecutive 'num' into a single 'num'

    Parameters
    ----------
    texts : iterable of str
        A list/iterable of texts.
    batch_size : int, optional
        Batch size for nlp.pipe (default=800).

    Returns
    -------
    tokens_list : list[list[str]]
        Tokens per text.
    lemmas_str_list : list[str]
        Same tokens joined into a single string per text.
    """
    tokens_list = []
    lemmas_str_list = []
    NEGATION_WORDS = {"not", "n't", "no", "never"}

    for doc in nlp.pipe(texts, batch_size=batch_size):
        toks = []
        for token in doc:
            if token.is_stop and token.lower_ not in NEGATION_WORDS:
                continue
            if token.is_punct:
                if token.text in ("!", "?"):
                    toks.append(token.text)
                continue
            toks.append(token.lemma_)

        new_toks = []
        prev_is_num = False
        for t in toks:
            if t == "num" and prev_is_num:
                continue
            new_toks.append(t)
            prev_is_num = (t == "num")

        tokens_list.append(new_toks)
        lemmas_str_list.append(" ".join(new_toks))

    return tokens_list, lemmas_str_list


def top_ngrams(texts, ngram_range=(1, 1), top_k=20, min_df=5):
    """
    Compute most frequent n-grams over a collection of texts.

    Parameters
    ----------
    texts : list[str] or iterable
        Text list to analyze.
    ngram_range : tuple(int, int), optional
        e.g., (1,1)=unigrams, (1,2)=unigrams+bigrams (default=(1,1))
    top_k : int, optional
        Number of top n-grams to return (default=20)
    min_df : int, optional
        Minimum frequency to include an n-gram (default=5)

    Returns
    -------
    list[tuple(str, int)]
        List of (ngram, frequency), sorted descending by frequency.
    """
    vec = CountVectorizer(ngram_range=ngram_range, min_df=min_df, lowercase=False)
    X = vec.fit_transform(texts)
    freqs = np.asarray(X.sum(axis=0)).ravel()
    feats = vec.get_feature_names_out()
    idx = np.argsort(freqs)[::-1][:top_k]
    return list(zip(feats[idx], freqs[idx]))


def plot_bar(items, title):
    """
    Horizontal bar plot for n-gram/word frequencies.

    Parameters
    ----------
    items : list[tuple(str,int)]
        List of (word/ngram, frequency)
    title : str
        Plot title

    Returns
    -------
    None
    """
    if not items:
        print(f"{title}: (κανένα n-gram με τα τωρινά φίλτρα)")
        return
    words, counts = zip(*items)
    plt.figure(figsize=(10, 5))
    plt.barh(words[::-1], counts[::-1])
    plt.title(title)
    plt.tight_layout()
    plt.show()


def vectorize_texts(df, feature_name, method="tfidf", model_path=None, max_features=15000):
    """
    Convert texts to vectors using TF-IDF or Word2Vec.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that holds the texts.
    feature_name : str
        Column name with text or tokens.
    method : str, optional
        'tfidf' (default) or 'word2vec'
    model_path : str or None
        Path to Word2Vec model (.model or .bin.gz) when method='word2vec'
    max_features : int, optional
        Max features for TF-IDF (default=15000)

    Returns
    -------
    TF-IDF:
        tuple (X, vectorizer)
        X : np.ndarray (float32)
        vectorizer : fitted TfidfVectorizer
    Word2Vec:
        X : np.ndarray (float32)
    """
    if method == "tfidf":
        print("Vectorizing with TF-IDF...")
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            lowercase=False,
            stop_words=None,
            ngram_range=(1, 2)
        )
        X = vectorizer.fit_transform(df[feature_name]).toarray().astype(np.float32, copy=False)
        return X, vectorizer

    elif method == "word2vec":
        if model_path is None:
            raise ValueError("Δώσε path στο Word2Vec μοντέλο (.model ή .bin.gz)")
        print(f"Loading Word2Vec model from: {model_path}")
        if model_path.endswith(".model"):
            model = KeyedVectors.load(model_path, mmap='r')
        else:
            model = KeyedVectors.load_word2vec_format(model_path, binary=True)

        print("Vectorizing with Word2Vec...")
        document_vectors = []
        for tokens in df[feature_name]:
            word_vecs = [model[w] for w in tokens if w in model]
            if word_vecs:
                document_vectors.append(np.mean(word_vecs, axis=0))
            else:
                document_vectors.append(np.zeros(model.vector_size))
        X = np.array(document_vectors, dtype=np.float32)
        return X

    else:
        raise ValueError("method πρέπει να είναι 'tfidf' ή 'word2vec'")


class ANN(nn.Module):
    """
    Simple feedforward neural network with 1 or 2 hidden layers.
    Supports BatchNorm or LayerNorm option.

    Parameters
    ----------
    input_dim : int
        Input dimensions
    output_dim : int
        Output dimensions (e.g., num classes)
    hidden_layer1 : int, optional
        Neurons in the first hidden layer (default=100)
    hidden_layer2 : int or None, optional
        Neurons in the second hidden layer (default=None → 1 hidden layer)
    dropoutp1 : float, optional
        Dropout probability for the first hidden layer (default=0.4)
    dropoutp2 : float, optional
        Dropout probability for the second hidden layer (default=0.4)
    norm : str or None, optional
        'batch' for BatchNorm, 'layer' for LayerNorm, None for no normalization (default=None)

    Forward Pass
    ------------
    With 1 hidden layer:
        input → Linear → [Norm] → ReLU → Dropout → Linear → output
    With 2 hidden layers:
        input → Linear → [Norm] → ReLU → Dropout → Linear → [Norm] → ReLU → Dropout → Linear → output
    """

    def __init__(self, input_dim, output_dim, hidden_layer1=100, hidden_layer2=None,
                 dropoutp1=0.4, dropoutp2=0.4, norm=None):
        super(ANN, self).__init__()
        self.norm = norm

        # First hidden layer
        self.fc1 = nn.Linear(input_dim, hidden_layer1)
        if norm == "batch":
            self.norm1 = nn.BatchNorm1d(hidden_layer1)
        elif norm == "layer":
            self.norm1 = nn.LayerNorm(hidden_layer1)

        # Second hidden layer (optional)
        if hidden_layer2 is None:
            self.fc2 = nn.Linear(hidden_layer1, output_dim)
        else:
            self.fc2 = nn.Linear(hidden_layer1, hidden_layer2)
            if norm == "batch":
                self.norm2 = nn.BatchNorm1d(hidden_layer2)
            elif norm == "layer":
                self.norm2 = nn.LayerNorm(hidden_layer2)
            self.fc3 = nn.Linear(hidden_layer2, output_dim)

        # Dropout layers
        self.dropout1 = nn.Dropout(p=dropoutp1)
        if hidden_layer2 is not None:
            self.dropout2 = nn.Dropout(p=dropoutp2)

    def forward(self, x):
        # First hidden
        x = self.fc1(x)
        if self.norm == "batch":
            x = self.norm1(x)
        elif self.norm == "layer":
            x = self.norm1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        # Second hidden (if present)
        if hasattr(self, "fc3"):
            x = self.fc2(x)
            if self.norm == "batch":
                x = self.norm2(x)
            elif self.norm == "layer":
                x = self.norm2(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc3(x)
        else:
            x = self.fc2(x)

        return x


def train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        num_epochs=20,
        patience=5,
        scheduler=None
):
    """
    Train a PyTorch model (CPU only) with early stopping based on validation loss.

    Args:
        model (torch.nn.Module): The neural network model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation set.
        criterion (torch.nn.Module): Loss function (e.g., CrossEntropyLoss).
        optimizer (torch.optim.Optimizer): Optimizer (e.g., Adam, SGD).
        num_epochs (int): Max number of epochs (default: 20).
        patience (int): Early stopping patience (default: 5).
        scheduler: Learning rate scheduler. If ReduceLROnPlateau, step with val loss.

    Returns:
        model, train_losses, val_losses, train_accuracies, val_accuracies
    """
    early_stopping = EarlyStopping(patience=patience, min_delta=0.0001)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    model.cpu()

    for epoch in range(num_epochs):
        # 1) Training
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_samples = 0

        for inputs, labels in train_loader:
            inputs = inputs.float()
            if isinstance(criterion, nn.CrossEntropyLoss):
                labels = labels.long()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, dim=1)
            correct_preds += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = correct_preds / total_samples if total_samples > 0 else 0.0
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)

        # 2) Validation
        model.eval()
        running_val_loss = 0.0
        correct_val_preds = 0
        val_samples = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.float()
                if isinstance(criterion, nn.CrossEntropyLoss):
                    labels = labels.long()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs, dim=1)
                correct_val_preds += (predicted == labels).sum().item()
                val_samples += labels.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)

        # Scheduler step
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(epoch_val_loss)
        elif scheduler is not None:
            scheduler.step()

        epoch_val_acc = correct_val_preds / val_samples if val_samples > 0 else 0.0
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | "
              f"Train Acc: {epoch_train_acc:.4f} | Val Acc: {epoch_val_acc:.4f}")

        # 3) Early stopping
        early_stopping(epoch_val_loss, model=model, epoch=epoch)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    if early_stopping.best_state_dict is not None:
        early_stopping.restore_best_weights(model)
        print(f"Restored best weights from epoch {early_stopping.best_epoch + 1} "
              f"(best val loss: {early_stopping.best_loss:.4f}).")

    return model, train_losses, val_losses, train_accuracies, val_accuracies


class EarlyStopping:
    """
    Early stopping watching validation loss with snapshotting of best weights.

    Logic:
    - On first call or any improvement of val_loss (> min_delta), take a snapshot of weights
    - If no improvement for 'patience' consecutive epochs, set early_stop=True
    - restore_best_weights(model) restores the best weights

    Parameters
    ----------
    patience : int, optional
        Number of consecutive epochs without improvement before stopping (default=5)
    min_delta : float, optional
        Minimum required improvement in val_loss to be considered better (default=0.0)

    Attributes
    ----------
    counter : int
        Count of epochs without improvement
    best_loss : float or None
        Best observed val_loss
    early_stop : bool
        Flag indicating if training should stop
    best_state_dict : dict or None
        Deepcopy of best model weights
    best_epoch : int
        Epoch index (0-based) where best val_loss occurred
    """

    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_state_dict = None
        self.best_epoch = -1

    def __call__(self, val_loss, model=None, epoch=None):
        # Snapshot on first call AND on any improvement
        if self.best_loss is None or val_loss < (self.best_loss - self.min_delta):
            self.best_loss = val_loss
            self.counter = 0
            if model is not None:
                self.best_state_dict = copy.deepcopy(model.state_dict())
            if epoch is not None:
                self.best_epoch = epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def restore_best_weights(self, model):
        if self.best_state_dict is None:
            print("No snapshot saved — check EarlyStopping call got model/epoch.")
            return
        model.load_state_dict(self.best_state_dict)


def plot_losses_to_pdf(
        train_losses,
        val_losses,
        pdf_filename='training_validation_losses.pdf',
        figsize=(8, 6),
        show_plot=False,
):
    """
    Plot training and validation losses and save to a PDF.

    Args:
        train_losses (list): Training loss per epoch.
        val_losses (list): Validation loss per epoch.
        pdf_filename (str): Target PDF file.
        figsize (tuple): Figure size (width, height).
        show_plot (bool): Whether to display the plot.

    Return:
        pdf_filename (str): The filename of the generated PDF.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(train_losses, label="Train Loss")
    ax.plot(val_losses, label="Val Loss")

    ax.set_title("Training & Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()

    with PdfPages(pdf_filename) as pdf:
        pdf.savefig(fig)

    if show_plot:
        plt.show()
    plt.close(fig)
    print(f"The learning curves are generated in pdf: {pdf_filename}")
    return pdf_filename


def get_preds(model, loader):
    """
    Extract ground-truth labels and model predictions from a DataLoader.

    Parameters
    ----------
    model : torch.nn.Module
        Trained PyTorch model.
    loader : torch.utils.data.DataLoader
        DataLoader for the dataset (val/test).

    Returns
    -------
    y_true : np.ndarray
        Ground-truth labels.
    y_pred : np.ndarray
        Predicted class indices.
    """
    model.eval()
    ys, yhat = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.float()
            preds = model(xb).argmax(dim=1)
            ys.append(yb.cpu().numpy())
            yhat.append(preds.cpu().numpy())
    return np.concatenate(ys), np.concatenate(yhat)