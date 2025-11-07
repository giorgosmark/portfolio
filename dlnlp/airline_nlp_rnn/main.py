"""
main.py — Main workflow for the text classification (sentiment analysis) project (BiLSTM + Attention)

Purpose:
    - Orchestrates the full NLP pipeline for airline tweet sentiment analysis using a BiLSTM
      with attention and pretrained Word2Vec embeddings.
    - Handles dataset loading, cleaning, tokenization/lemmatization, sequence building, model
      training with early stopping, and final evaluation/visualization.
    - Keeps reusable logic in utils.py to maintain a clean, modular structure.

Workflow Overview:
    1) Load the dataset (Tweets.csv) via load_data() and perform initial inspection (info, NaNs, head, stats).
    2) Select relevant columns (text, airline_sentiment) and label-encode sentiments to integer ids.
    3) Clean raw text (HTML unescape/removal, strip mentions/URLs/hashtags, noise reduction, casing/spacing normalization).
    4) Tokenize & lemmatize with spaCy in batches (keep '!' and '?' only; compress consecutive 'num').
    5) Remove rows with invalid/empty token lists (post-processing hygiene).
    6) Inspect token-length distribution and percentiles to choose a fixed MAX_LEN for padding/truncation.
    7) Build the vocabulary (sorted unique tokens) and reserve special ids: <PAD>=0, <UNK>=1.
    8) Convert each tweet’s tokens to fixed-length id sequences (truncate/pad to MAX_LEN).
    9) Load the pretrained Word2Vec model (update `w2v_path` to your local path) and construct
       a trainable nn.Embedding aligned to the vocabulary (PAD=zero row, UNK=mean vector).
   10) Split data into Train/Validation/Test (70/15/15) with stratification.
   11) Create PyTorch TensorDatasets/DataLoaders (ids as torch.long for Embedding; labels as torch.long for CE loss).
   12) Define the BiLSTM+Attention model (residual projection; PAD-masked attention), loss (CrossEntropy), optimizer (Adam),
       and scheduler (linear LR decay) with early stopping (patience).
   13) Train the model and log epoch-wise Train/Val loss & accuracy; restore best weights on early stop.
   14) Export training/validation loss curves to PDF for documentation.
   15) Evaluate on the test set (overall accuracy, classification report) and visualize confusion matrices
       (raw counts & row-normalized).

Notes:
    - All reusable utilities (preprocessing functions, model class, training loop, plotting helpers)
      live in utils.py and are imported here.
    - Adjust dataset path in load_data() (e.g., datasets/Tweets.csv).
    - Update the Word2Vec model path (`w2v_path`) to match your local filesystem:
      Example: r"C:\\Users\\<username>\\Projects\\word2vec-google-news-300.model"
    - The pipeline runs on CPU by design (no GPU required).
    - Any notebook `display(...)` calls have been replaced with `print(...)` equivalents where applicable.
    - pandas display options and plotting styles (if any) should be configured locally as needed.

Maintenance:
    - For new experiments (e.g., different RNN sizes/layers, alternative schedulers, TF-IDF baselines,
      or transformer-based models), keep preprocessing and helper logic in utils.py and let main.py
      focus on the high-level workflow orchestration.
"""

# main.py

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from gensim.models import KeyedVectors

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# from sklearn.utils.class_weight import compute_class_weight  # kept as in the notebook

from utils import (
    load_data,
    inspect_data,
    clean_text,
    spacy_preprocess_batch,
    tokens_to_padded_ids_simple,
    build_embedding_layer,
    TwoLayerBiLSTMAttention,
    train_model,
    plot_losses_to_pdf,
    get_preds,
    plot_confusion_matrices
)

# --------------------------------------------------------------------
# 1) Load raw data
# --------------------------------------------------------------------
df, _ = load_data(source='file', filepath='datasets/Tweets.csv', dataset_func=None, sheet_name=None)

# --------------------------------------------------------------------
# 2) Quick inspection
# --------------------------------------------------------------------
inspect_data(df, target_column=None)

# --------------------------------------------------------------------
# 3) Keep only relevant columns (features/target) and drop missing rows
# --------------------------------------------------------------------
df = df[["text", "airline_sentiment"]].dropna()

# --------------------------------------------------------------------
# 4) Label encoding for the target column (string -> integer ids)
# --------------------------------------------------------------------
le = LabelEncoder()
df["label"] = le.fit_transform(df["airline_sentiment"])
print("Classes (με τη σειρά των ids):", list(le.classes_))
print("Αριθμός κλάσεων:", len(le.classes_))


print(df["text"].head(-5))

# --------------------------------------------------------------------
# 5) Text cleaning step (keeps original flags/logic)
# --------------------------------------------------------------------
df.loc[:, "cleaned_text"] = df["text"].apply(clean_text)

print("\nΚαθαρισμένα κείμενα (preview):")
print(df["cleaned_text"].head(-5))

# --------------------------------------------------------------------
# 6) Tokenization & Lemmatization (spaCy) in batches
#    Returns both token lists and a space-joined lemma string if needed.
# --------------------------------------------------------------------
tokens_col, lemmas_col = spacy_preprocess_batch(df["cleaned_text"].tolist(), batch_size=800)
df["tokens"] = tokens_col
df["lemmas_str"] = lemmas_col

print("\nTokens (preview):")
print(df["tokens"].head(-5))

# --------------------------------------------------------------------
# 7) Remove rows with NaN/empty tokens (post-processing hygiene)
#    Many NLP vectorizers choke on empty inputs
# --------------------------------------------------------------------
mask_tokens_nan   = df["tokens"].isna()
mask_tokens_empty = df["tokens"].map(lambda x: len(x) == 0)
print("NaN tokens:",   mask_tokens_nan.sum())
print("Empty tokens:", mask_tokens_empty.sum())

bad_mask = mask_tokens_nan | mask_tokens_empty
print("\nΣύνολο που θα αφαιρεθούν:", bad_mask.sum())
df = df.loc[~bad_mask].reset_index(drop=True)
print("Νέο μέγεθος df:", len(df))

# Confirm post-clean counts
mask_tokens_nan   = df["tokens"].isna()
mask_tokens_empty = df["tokens"].map(lambda x: len(x) == 0)
print("NaN tokens:",   mask_tokens_nan.sum())
print("Empty tokens:", mask_tokens_empty.sum())

# --------------------------------------------------------------------
# 8) Inspect token length distribution to choose a fixed MAX_LEN
# --------------------------------------------------------------------
lengths = df["tokens"].map(len)
print("\nΠεριγραφή μηκών (tokens ανά tweet):")
print(lengths.describe())
print("Percentiles 50/90/95/99:",
      np.percentile(lengths, [50, 90, 95, 99]))

# --------------------------------------------------------------------
# 9) Build vocabulary (token -> id) with special PAD/UNK reserved
# --------------------------------------------------------------------
unique_tokens = {tok for toks in df["tokens"] for tok in toks}
print("Μοναδικά tokens (σύνολο):", len(unique_tokens))

# Sorted for stable ids across runs
all_tokens = sorted(unique_tokens)

word2id = {"<PAD>": 0, "<UNK>": 1}
for i, w in enumerate(all_tokens, start=2):
    word2id[w] = i
id2word = {i: w for w, i in word2id.items()}

print("vocab_size (με PAD/UNK):", len(word2id))
print("Πρώτα 10 ζευγάρια:", list(word2id.items())[:10])

# --------------------------------------------------------------------
# 10) Fixed constants for RNN sequence modeling
# --------------------------------------------------------------------
PAD_ID = 0
UNK_ID = 1
MAX_LEN = 35
print(PAD_ID, UNK_ID, MAX_LEN)

# --------------------------------------------------------------------
# 11) Convert tokenized tweets to fixed-length id sequences
#     Keep dtype=int64 for PyTorch Embedding compatibility.
# --------------------------------------------------------------------
rows = []
for toks in df["tokens"]:
    row_ids = tokens_to_padded_ids_simple(
        tokens=toks,
        word2id=word2id,
        MAX_LEN=MAX_LEN,
        PAD_ID=PAD_ID,
        UNK_ID=UNK_ID
    )
    rows.append(row_ids)

X_seq = np.array(rows, dtype=np.int64)
y_seq = df["label"].to_numpy(dtype=np.int64)
labels = df["label"].to_numpy(dtype=np.int64)
num_classes = len(np.unique(labels))

print("X_seq shape:", X_seq.shape)
print("y_seq shape:", y_seq.shape)
print("Παράδειγμα 1ης σειράς ids:", X_seq[0].tolist())

print(id2word[7586])
print(num_classes)

# --------------------------------------------------------------------
# 12) Load pretrained Word2Vec
# --------------------------------------------------------------------
w2v_path = r"C:\Users\giorg\PycharmProjects\PythonProject4\word2vec-google-news-300.model"
w2v_model = KeyedVectors.load(w2v_path, mmap='r')
print("Vector size:", w2v_model.vector_size)

# --------------------------------------------------------------------
# 13) Build embedding layer aligned to our vocabulary, with PAD/UNK rows
# --------------------------------------------------------------------
embedding_layer_seq = build_embedding_layer(word2id, w2v_model)
print(embedding_layer_seq)

# --------------------------------------------------------------------
# 14) Split into Train/Validation/Test with stratification
# --------------------------------------------------------------------
train_X_seq, temp_X_seq, train_y_seq, temp_y_seq = train_test_split(
    X_seq, y_seq,
    test_size=0.20,
    random_state=42,
    stratify=y_seq
)

val_X_seq, test_X_seq, val_y_seq, test_y_seq = train_test_split(
    temp_X_seq, temp_y_seq,
    test_size=0.50,
    random_state=42,
    stratify=temp_y_seq
)

print("Train:", train_X_seq.shape, train_y_seq.shape)
print("Val:  ", val_X_seq.shape,   val_y_seq.shape)
print("Test: ", test_X_seq.shape,  test_y_seq.shape)

# --------------------------------------------------------------------
# 15) Wrap arrays into PyTorch Tensors and DataLoaders
#     Keep long dtype for ids; CrossEntropy expects long labels too.
# --------------------------------------------------------------------
train_X_tensor_seq = torch.tensor(train_X_seq, dtype=torch.long)
train_y_tensor_seq = torch.tensor(train_y_seq, dtype=torch.long)

val_X_tensor_seq   = torch.tensor(val_X_seq,   dtype=torch.long)
val_y_tensor_seq   = torch.tensor(val_y_seq,   dtype=torch.long)

test_X_tensor_seq  = torch.tensor(test_X_seq,  dtype=torch.long)
test_y_tensor_seq  = torch.tensor(test_y_seq,  dtype=torch.long)

train_dataset_seq = TensorDataset(train_X_tensor_seq, train_y_tensor_seq)
val_dataset_seq   = TensorDataset(val_X_tensor_seq,   val_y_tensor_seq)
test_dataset_seq  = TensorDataset(test_X_tensor_seq,  test_y_tensor_seq)

batch_size = 32
train_loader_seq = DataLoader(train_dataset_seq, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader_seq   = DataLoader(val_dataset_seq,   batch_size=batch_size, shuffle=False, num_workers=0)
test_loader_seq  = DataLoader(test_dataset_seq,  batch_size=batch_size, shuffle=False, num_workers=0)

print("OK: έτοιμα train/val/test loaders για SEQ (ids).")

# --------------------------------------------------------------------
# 16) Define the BiLSTM + Attention model using the pretrained Embedding
# --------------------------------------------------------------------
model_rnn_w2v = TwoLayerBiLSTMAttention(
    input_dim=embedding_layer_seq.embedding_dim,  # unused when embedding is provided
    hidden_dim=128,
    output_dim=num_classes,
    num_layers=2,
    dropout=0.3,
    bidirectional=True,
    embedding_layer=embedding_layer_seq
)

# --------------------------------------------------------------------
# 17) Loss & Optimizer (class weights scaffold commented)
# --------------------------------------------------------------------
# classes = np.unique(train_y)
# cls_weights = compute_class_weight('balanced', classes=classes, y=train_y)
# cls_weights = torch.tensor(cls_weights, dtype=torch.float32)
# loss_function = nn.CrossEntropyLoss(weight=cls_weights)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_rnn_w2v.parameters(), lr=2e-4)

# --------------------------------------------------------------------
# 18) Scheduler & early stopping parameters (linear decay here)
# --------------------------------------------------------------------
num_epochs = 200
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda e: max(0.0, 1.0 - e / float(num_epochs-1))
)
# Alternative kept from the notebook:
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6)
patience = 5

# --------------------------------------------------------------------
# 19) Train the model with validation monitoring & early stopping
# --------------------------------------------------------------------
model_rnn_w2v, tr_losses, val_losses, tr_accs, val_accs = train_model(
    model=model_rnn_w2v,
    train_loader=train_loader_seq,
    val_loader=val_loader_seq,
    criterion=loss_function,
    optimizer=optimizer,
    num_epochs=num_epochs,
    patience=patience,
    scheduler=scheduler
)

# --------------------------------------------------------------------
# 20) Plot training curves to PDF (same filename as in the notebook)
# --------------------------------------------------------------------
plot_losses_to_pdf(train_losses=tr_losses,
                   val_losses=val_losses,
                   pdf_filename="training_validation_losses.pdf",
                   figsize=(12, 8),
                   show_plot=True)

# --------------------------------------------------------------------
# 21) Evaluate on the test split
# --------------------------------------------------------------------
model_rnn_w2v.eval()
correct_w2v, total_w2v = 0, 0
with torch.no_grad():
    for xb, yb in test_loader_seq:
        preds = model_rnn_w2v(xb).argmax(1)
        correct_w2v += (preds == yb).sum().item()
        total_w2v   += yb.size(0)

test_acc_w2v = 100.0 * correct_w2v / total_w2v
print(f"Test Accuracy (W2V): {test_acc_w2v:.2f}%")

# --------------------------------------------------------------------
# 22) Classification report with per-class precision/recall/F1
# --------------------------------------------------------------------
y_true_w2v, y_pred_w2v = get_preds(model_rnn_w2v, test_loader_seq)
print(classification_report(y_true_w2v, y_pred_w2v, digits=3, zero_division=0))

# --------------------------------------------------------------------
# 23) Confusion matrices (counts and row-normalized)
# --------------------------------------------------------------------
y_true_w2v, y_pred_w2v = get_preds(model_rnn_w2v, test_loader_seq)
plot_confusion_matrices(y_true_w2v, y_pred_w2v, ["neg", "neu", "pos"], title_prefix="W2V")
