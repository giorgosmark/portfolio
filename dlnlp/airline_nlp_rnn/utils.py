"""
utils.py — Helper functions for the text classification (NLP) project (BiLSTM + Attention)

Purpose
    - Centralizes ALL reusable utilities for data loading/inspection, text preprocessing,
      sequence preparation, embedding construction, model definition/training, and evaluation.
    - Keeps main.py focused on the execution workflow (data paths, splits, hyperparameters, I/O).

Contents (indicative)
    - Data I/O & inspection:
        load_data, inspect_data
    - Emoji tools:
        EMOJI_RE, EMO_POS/EMO_NEG/EMO_NEU, map_emojis_to_tokens
    - Text preprocessing:
        clean_text, spacy_preprocess_batch
    - Sequence prep:
        tokens_to_padded_ids_simple
    - Embeddings:
        build_embedding_layer (Word2Vec-aligned nn.Embedding with PAD/UNK handling)
    - Model:
        TwoLayerBiLSTMAttention (BiLSTM with residual projection + lightweight attention)
    - Training:
        train_model (CPU-only loop with early stopping & optional LR scheduler),
        EarlyStopping (snapshot & restore best weights)
    - Evaluation & visualization:
        get_preds, plot_confusion_matrices, plot_losses_to_pdf

Usage
    from utils import (
        load_data, inspect_data,
        map_emojis_to_tokens, clean_text, spacy_preprocess_batch,
        tokens_to_padded_ids_simple, build_embedding_layer,
        TwoLayerBiLSTMAttention, train_model, EarlyStopping,
        get_preds, plot_confusion_matrices, plot_losses_to_pdf
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

# utils.py

import re
import copy
import html
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Sklearn
from sklearn.metrics import confusion_matrix

# NLP
import spacy
from gensim.models import KeyedVectors

# For exporting plots to PDF
from matplotlib.backends.backend_pdf import PdfPages

# ---------------------------------------------------------------------
# Load spaCy English model once (keeps tagger + lemmatizer for pipeline)
# Note: loading here mirrors the notebook's "global nlp" behavior.
# ---------------------------------------------------------------------
nlp = spacy.load("en_core_web_sm")


# ---------------------------------------------------------------------
# Emoji regex and sentiment buckets
# These support optional mapping to <emo_pos>/<emo_neg>/<emo_neu> tokens.
# ---------------------------------------------------------------------
EMOJI_RE = re.compile(
    "["                            # Unicode ranges for emojis
    u"\U0001F600-\U0001F64F"              # emoticons
    u"\U0001F300-\U0001F5FF"              # symbols & pictographs
    u"\U0001F680-\U0001F6FF"              # transport & map
    u"\U0001F1E0-\U0001F1FF"              # flags
    u"\U00002700-\U000027BF"              # misc symbols
    u"\U0001F900-\U0001F9FF"              # supplemental symbols
    "]+", flags=re.UNICODE
)

EMO_POS = set("😀😃😄😁🙂😊🥰😍😘😎🤩👍👏🎉✨💯❤️💕💖💙💚💛💜💗💓💞🤗👌")
EMO_NEG = set("😞😟😠😡😔😢😭😫🤬👎💔🙁☹️")
EMO_NEU = set("😐😑🤔😶🤨😴😕")


def map_emojis_to_tokens(s):
    """
    Replace emojis in text with semantic tokens
    (<emo_pos>, <emo_neg>, <emo_neu>) based on sentiment.

    Parameters
    ----------
    s : str
        Input text containing characters and emojis.

    Returns
    -------
    str
        Text where emojis have been substituted with tokens.
    """
    # Iterate char-by-char to detect emojis and inject sentiment placeholders.
    out = []
    for ch in s:
        if ch in EMO_POS:
            out.append(" <emo_pos> ")
        elif ch in EMO_NEG:
            out.append(" <emo_neg> ")
        elif ch in EMO_NEU:
            out.append(" <emo_neu> ")
        else:
            out.append(ch)
    return "".join(out)


def clean_text(text,
               clean_html=True,
               remove_amp=True,
               remove_mentions=True,
               remove_urls=True,
               strip_hashtags=True,
               remove_numbers=True,
               normalize_hyphens=True,
               map_emojis=False,
               remove_emojis=True,
               remove_noise=True,
               normalize_intensity=True,
               to_lower=True,
               normalize_space=True):
    """
    Text cleaning/normalization via modular flags. Useful for NLP/ML preprocessing.

    What it does (depending on flags)
    ---------------------------------
    - HTML: unescape entities and remove HTML tags (keeps patterns like '<3').
    - Replacements: normalize apostrophe (’→'), remove '&', '@mentions', URLs,
      strip '#' (keep the word), '_'→space.
    - Noise: remove ASCII emoticons (standalone), currency symbols, artifacts
      {…}, […], <…> and special characters (emoji/basic punctuation kept).
    - Numbers: replace digits with 'NUM' and compress multiple 'NUM' → one.
    - Intensity: compress sequences of '!?', reduce repeated '!'/'?' to one,
      and reduce 3+ repeated letters to 2.
    - Fixes: known artifacts ('flightled'/'flightle'/'flighted'/'flightr'→'flight',
      'rebooke'→'rebook').
    - Slang: thx/thanks→thank, pls/plz→please, u→you, ur→your, ppl→people.
    - Optional: map emojis to tokens, lowercasing, whitespace normalization.

    Parameters
    ----------
    text : str
        Raw input text.
    clean_html, remove_amp, remove_mentions, remove_urls, strip_hashtags,
    remove_numbers, normalize_hyphens, remove_noise, normalize_intensity,
    to_lower, normalize_space : bool, optional
        Enable/disable steps (default: True).
    map_emojis : bool, optional
        If True, applies `map_emojis_to_tokens` (default: False).

    Returns
    -------
    str
        Cleaned/normalized text.

    Notes
    -----
    Requires `re`, `html`, and `EMOJI_RE`. If `map_emojis=True`, requires
    `map_emojis_to_tokens`.
    """
    # Start with raw string and progressively normalize.
    s = text

    if clean_html:
        s = html.unescape(s)
        # Remove HTML-like tags but keep patterns such as "<3"
        s = re.sub(r"</?[A-Za-z!][^>]*>", "", s)

    # Apostrophe normalization helps lemmatization/tokenization consistency.
    s = s.replace("’", "'")

    if remove_amp:
        s = s.replace("&", " ")

    if remove_mentions:
        s = re.sub(r"(?<!\w)@\w+", "", s)

    if remove_urls:
        s = re.sub(r"(https?://\S+|www\.\S+)", "", s)

    if strip_hashtags:
        # Drop '#' but keep the token itself: #delay -> delay
        s = re.sub(r"#(\w+)", r"\1", s)
        # Remove standalone ASCII emoticons and currency symbols
        s = re.sub(r'(?i)(?<!\w)(?:[:;=x][\-^]?[)D\(P/\\]|<3)(?!\w)', ' ', s)
        s = re.sub(r'[$€£¥₩₽₹¢฿]', ' ', s)

    if remove_emojis:
        # If you later want semantic mapping, toggle map_emojis=True instead.
        s = EMOJI_RE.sub(" ", s)

    if remove_numbers:
        # Use NUM sentinel to avoid losing structure for models that can leverage it.
        s = re.sub(r"\d+", "NUM", s)
        s = re.sub(r"\bNUM(?:\W+NUM)+\b", "NUM", s)

    if normalize_hyphens:
        s = s.replace("_", " ")

    if remove_noise:
        # Strip annotations/artifacts and special typography
        s = re.sub(r"\{.*?\}|\[.*?\]|<.*?>", " ", s)
        s = re.sub(r"[©®™★☆•…–—·※“”\"]", " ", s)

    if normalize_intensity:
        # Compress '!?', long runs of '!'/'?', and elongated words
        s = re.sub(r'(?=[!?]*!)(?=[!?]*\?)[!?]{2,}', '!?', s)
        s = re.sub(r'!{2,}', '!', s)
        s = re.sub(r'\?{2,}', '?', s)
        s = re.sub(r'(.)\1{2,}', r'\1\1', s)

    # Domain-specific scrape/ocr fixes (kept as-is)
    s = re.sub(r"\bflight(?:led|le|ed|r)\b", "flight", s, flags=re.IGNORECASE)
    s = re.sub(r"\brebooke\b", "rebook", s, flags=re.IGNORECASE)

    # Lightweight slang normalization to reduce sparsity
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

    if map_emojis:
        s = map_emojis_to_tokens(s)

    if to_lower:
        s = s.lower()

    # Normalize multiple 'num' after lowercasing
    s = re.sub(r"\bnum(?:\W+num)+\b", "num", s)

    # Trim punctuation glued to token edges; keep '!' and '?' for sentiment
    s = re.sub(r"(?<=\w)[\.,:;]+(?=\s|$)", "", s)
    s = re.sub(r"(^|(?<=\s))[\.,:;]+(?=\w)", "", s)

    if normalize_space:
        s = re.sub(r"\s+", " ", s).strip()

    return s


def spacy_preprocess_batch(texts, batch_size=800):
    """
    Fast text preprocessing using spaCy in batches via nlp.pipe.

    Steps:
    - Tokenization & lemmatization with nlp.pipe
    - Stopwords are NOT removed (kept); negations can be retained if filtering is enabled
    - Remove punctuation except '!' and '?'
    - Use lemma instead of raw token
    - Normalize multiple 'num' to a single 'num'

    Parameters
    ----------
    texts : iterable of str
        List/iterable of texts.
    batch_size : int, optional
        Pipe batch size (default=800).

    Returns
    -------
    tokens_list : list[list[str]]
        Tokens per text (lemmas).
    lemmas_str_list : list[str]
        Space-joined lemmas per text.
    """
    # Use spaCy pipeline over batches for speed; mirrors notebook behavior.
    tokens_list = []
    lemmas_str_list = []

    for doc in nlp.pipe(texts, batch_size=batch_size):
        toks = []
        for token in doc:
            # Keep only '!' and '?' from punctuation; drop the rest.
            if token.is_punct:
                if token.text in ("!", "?"):
                    toks.append(token.text)
                continue
            # Prefer lemma to reduce sparsity (e.g., delayed/delays -> delay)
            toks.append(token.lemma_)

        # Compress consecutive 'num' placeholders: num num -> num
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


class TwoLayerBiLSTMAttention(nn.Module):
    """
    Two-layer (default) BiLSTM with attention and residual projection
    from input to LSTM output. Supports optional embedding layer with
    `padding_idx` for PAD masking.

    Parameters
    ----------
    input_dim : int
        Input dimension when no embedding is given (e.g., features per token).
    hidden_dim : int
        LSTM hidden size (per direction).
    output_dim : int
        Output dimension (e.g., number of classes).
    num_layers : int, optional
        Number of stacked LSTM layers (default=2).
    dropout : float, optional
        Dropout between LSTM layers (default=0.2).
    bidirectional : bool, optional
        Whether LSTM is bidirectional (default=True).
    embedding_layer : torch.nn.Embedding or None, optional
        External embedding for token ids. If provided with `padding_idx`,
        PAD positions are masked in attention.

    Notes
    -----
    - If `embedding_layer` is provided, forward expects LongTensor ids
      shaped (batch, seq_len). Otherwise expects FloatTensor embeddings
      shaped (batch, seq_len, input_dim).
    - Residual projection (`res_proj`) maps input to LSTM output dim and
      is added before attention.
    """
    def __init__(self, input_dim, hidden_dim, output_dim,
                 num_layers=2, dropout=0.2, bidirectional=True,
                 embedding_layer=None):
        super(TwoLayerBiLSTMAttention, self).__init__()

        # Persist configuration
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.embedding = embedding_layer
        self.pad_idx = embedding_layer.padding_idx if embedding_layer is not None else None

        # LSTM input width depends on whether we receive ids (via embedding) or vectors
        lstm_input_dim = (self.embedding.embedding_dim if self.embedding is not None else input_dim)

        # Stacked (possibly bidirectional) LSTM
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )

        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # Residual projection aligns input width to LSTM output width
        self.res_proj = nn.Linear(lstm_input_dim, lstm_output_dim)

        # Lightweight additive attention (MLP -> scalar energy per timestep)
        self.attn = nn.Sequential(
            nn.Linear(lstm_output_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Final classifier over the aggregated context
        self.fc = nn.Linear(lstm_output_dim, output_dim)

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            With embedding: LongTensor of token ids (batch, seq_len).
            Without embedding: FloatTensor embeddings (batch, seq_len, input_dim).

        Returns
        -------
        torch.Tensor
            Logits of shape (batch_size, output_dim).

        Notes
        -----
        - Attention via MLP (lstm_out -> 64 -> 1), softmax over time (dim=1).
          If `padding_idx` is set, PAD tokens are masked before softmax.
        - Context vector is weighted sum over timesteps, then projected by `fc`.
        """
        # Keep original ids to build attention mask if padding_idx is set
        token_ids = x if (self.embedding is not None and self.pad_idx is not None) else None

        if self.embedding is not None:
            x = self.embedding(x)

        # Residual path uses the exact embedding input to the LSTM
        x_in = x
        lstm_out, _ = self.lstm(x)
        y = lstm_out + self.res_proj(x_in)

        energy = self.attn(y)

        if token_ids is not None:
            # Mask PAD positions so they do not attract attention mass
            mask = (token_ids != self.pad_idx).unsqueeze(-1)
            energy = energy.masked_fill(~mask, -1e9)

        attention_weights = F.softmax(energy, dim=1)
        context = (y * attention_weights).sum(dim=1)
        out = self.fc(context)
        return out


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
    Train a PyTorch model (CPU only) with early stopping on validation loss.
    Logs train/val loss & accuracy per epoch and optionally steps a LR scheduler.

    Parameters
    ----------
    model : torch.nn.Module
    train_loader : torch.utils.data.DataLoader
    val_loader : torch.utils.data.DataLoader
    criterion : torch.nn.Module
    optimizer : torch.optim.Optimizer
    num_epochs : int, optional
    patience : int, optional
    scheduler : torch.optim.lr_scheduler._LRScheduler or ReduceLROnPlateau or None, optional

    Returns
    -------
    model : torch.nn.Module
        Trained model with best weights restored (if found).
    train_losses : list[float]
    val_losses : list[float]
    train_accuracies : list[float]
    val_accuracies : list[float]

    Notes
    -----
    - If criterion is CrossEntropyLoss, labels are cast to LongTensor.
    - If scheduler is ReduceLROnPlateau, calls `scheduler.step(val_loss)`; else `scheduler.step()`.
    """
    # Early stopping tracks best val_loss and keeps a snapshot of weights
    early_stopping = EarlyStopping(patience=patience, min_delta=0.0001)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    # Keep everything on CPU as in the notebook
    model.cpu()

    for epoch in range(num_epochs):
        # -----------------------
        # 1) Training epoch
        # -----------------------
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_samples = 0

        for inputs, labels in train_loader:
            if isinstance(criterion, nn.CrossEntropyLoss):
                labels = labels.long()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # Optional gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, dim=1)
            correct_preds += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = correct_preds / total_samples if total_samples > 0 else 0.0
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)

        # -----------------------
        # 2) Validation epoch
        # -----------------------
        model.eval()
        running_val_loss = 0.0
        correct_val_preds = 0
        val_samples = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                if isinstance(criterion, nn.CrossEntropyLoss):
                    labels = labels.long()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs, dim=1)
                correct_val_preds += (predicted == labels).sum().item()
                val_samples += labels.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)

        # Step LR scheduler according to notebook logic
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(epoch_val_loss)
        else:
            scheduler.step()

        epoch_val_acc = correct_val_preds / val_samples if val_samples > 0 else 0.0
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | "
              f"Train Acc: {epoch_train_acc:.4f} | Val Acc: {epoch_val_acc:.4f}")

        # -----------------------
        # 3) Early stopping check
        # -----------------------
        early_stopping(epoch_val_loss, model=model, epoch=epoch)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # Restore best weights if a snapshot exists
    if early_stopping.best_state_dict is not None:
        early_stopping.restore_best_weights(model)
        print(f"Restored best weights from epoch {early_stopping.best_epoch + 1} "
              f"(best val loss: {early_stopping.best_loss:.4f}).")

    return model, train_losses, val_losses, train_accuracies, val_accuracies


class EarlyStopping:
    """
    Early stopping monitoring validation loss with snapshot of best weights.

    Logic:
    - On first call or whenever val_loss improves by > min_delta: save snapshot
    - If no improvement for 'patience' consecutive epochs: early_stop=True
    - `restore_best_weights(model)` restores the best weights

    Parameters
    ----------
    patience : int, optional
        Number of epochs without improvement before stopping (default=5)
    min_delta : float, optional
        Minimal improvement required to be considered an improvement (default=0.0)

    Attributes
    ----------
    counter : int
        Epochs without improvement
    best_loss : float or None
        Best val_loss observed
    early_stop : bool
        Whether to stop training
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
        # Save snapshot on first call or on sufficient improvement
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
    Plot training and validation losses and save figure to a PDF file.

    Args:
        train_losses (list): Training loss per epoch.
        val_losses (list): Validation loss per epoch.
        pdf_filename (str): Output PDF filename.
        figsize (tuple): Figure size (width, height).
        show_plot (bool): Whether to display the plot.

    Return:
        pdf_filename (str): The generated PDF filename.
    """
    # One-figure, two-lines plot (train vs val)
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
        DataLoader for val/test.

    Returns
    -------
    y_true : np.ndarray
        Ground-truth labels.
    y_pred : np.ndarray
        Predicted class indices.
    """
    # Disable dropout etc. for deterministic inference
    model.eval()
    ys, yhat = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds = model(xb).argmax(dim=1)
            ys.append(yb.cpu().numpy())
            yhat.append(preds.cpu().numpy())
    return np.concatenate(ys), np.concatenate(yhat)


def tokens_to_padded_ids_simple(tokens, word2id, MAX_LEN, PAD_ID, UNK_ID):
    """
    Convert a list of tokens into a fixed-length list of ids.

    Parameters
    ----------
    tokens : list of str
        Input tokens.
    word2id : dict
        Mapping word -> id.
    MAX_LEN : int
        Maximum sequence length.
    PAD_ID : int
        Padding id.
    UNK_ID : int
        Unknown-word id.

    Returns
    -------
    ids : list of int
        List of ids with exactly length MAX_LEN.
    """
    # Map tokens to ids with UNK fallback, then truncate/pad to fixed width.
    ids = []
    for t in tokens:
        if t in word2id:
            ids.append(word2id[t])
        else:
            ids.append(UNK_ID)

    if len(ids) > MAX_LEN:
        ids = ids[:MAX_LEN]

    while len(ids) < MAX_LEN:
        ids.append(PAD_ID)

    return ids


def plot_confusion_matrices(y_true, y_pred, class_names, title_prefix="Model"):
    """
    Plot side-by-side confusion matrices:
    - Counts (absolute numbers)
    - Normalized (row-wise percentages)

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Predicted labels.
    class_names : list of str
        Class names for axes.
    title_prefix : str, optional
        Prefix for plot titles.

    Returns
    -------
    None
    """
    # Prepare label indices and compute confusion matrices
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    # Side-by-side layout for counts vs normalized heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    im1 = axes[0].imshow(cm, cmap="Blues")
    axes[0].set_xticks(labels); axes[0].set_yticks(labels)
    axes[0].set_xticklabels(class_names); axes[0].set_yticklabels(class_names)
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")
    axes[0].set_title(f"{title_prefix} (counts)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[0].text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(cm_norm, cmap="Blues")
    axes[1].set_xticks(labels); axes[1].set_yticks(labels)
    axes[1].set_xticklabels(class_names); axes[1].set_yticklabels(class_names)
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")
    axes[1].set_title(f"{title_prefix} (normalized)")
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            axes[1].text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", color="black")
    fig.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.show()


def build_embedding_layer(word2id, w2v_model, PAD_ID=0, UNK_ID=1):
    """
    Create an embedding matrix (using Word2Vec where available) and wrap it in a
    PyTorch Embedding layer.

    Parameters
    ----------
    word2id : dict
        Token -> id mapping.
    w2v_model : gensim KeyedVectors
        Pretrained Word2Vec vectors.
    PAD_ID : int, optional
        PAD token index (default=0).
    UNK_ID : int, optional
        UNK token index (default=1).

    Returns
    -------
    embedding_layer : nn.Embedding
        Ready-to-use PyTorch embedding layer.
    """
    # Derive vocabulary size and vector dimensionality from inputs
    vocab_size = len(word2id)
    emb_dim = w2v_model.vector_size

    # Initialize with small Gaussian noise as a baseline
    emb_matrix = np.random.normal(0, 0.01, size=(vocab_size, emb_dim)).astype(np.float32)

    # PAD row is zero so it does not contribute in aggregations
    emb_matrix[PAD_ID] = np.zeros(emb_dim, dtype=np.float32)

    # Fill with Word2Vec vectors where available
    filled = []
    for w, idx in word2id.items():
        if idx < 2:   # Skip PAD/UNK
            continue
        if w in w2v_model:
            emb_matrix[idx] = w2v_model[w]
            filled.append(idx)

    # UNK row becomes the mean of all known vectors (neutral representation)
    if filled:
        emb_matrix[UNK_ID] = emb_matrix[filled].mean(axis=0)
    else:
        emb_matrix[UNK_ID] = np.random.normal(0, 0.01, size=(emb_dim,)).astype(np.float32)

    # Wrap as a trainable nn.Embedding with padding_idx set
    embedding_layer = nn.Embedding(
        num_embeddings=vocab_size,
        embedding_dim=emb_dim,
        padding_idx=PAD_ID
    )
    embedding_layer.weight.data.copy_(torch.from_numpy(emb_matrix))
    embedding_layer.weight.requires_grad = True

    print("Embedding layer έτοιμο:", embedding_layer.num_embeddings, "x", embedding_layer.embedding_dim)
    return embedding_layer


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