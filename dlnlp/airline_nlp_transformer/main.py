"""
    main.py — Main workflow for the Transformer-based sentiment classification project (BERT focus)

    Purpose:
        - Orchestrates the full NLP pipeline for airline tweet sentiment classification using
        Hugging Face Transformers (BERT-based models).
        - Handles dataset loading, label encoding, tokenization, model initialization, fine-tuning
        (classifier-only and last-layer unfreeze), evaluation, and visualization.
        - Keeps reusable logic in utils.py to maintain a modular and organized structure.

    Workflow Overview:
        1) Mount Google Drive and set up the working directory (Colab environment).
        2) Install required dependencies dynamically in Colab using pip commands.
        3) Load dataset (Tweets.csv) via load_data() and inspect structure, NaNs, and class balance.
        4) Keep only text and sentiment columns, drop duplicates, and label-encode sentiments.
        5) Split data into Train/Validation/Test sets (80/10/10 stratified by label).
        6) Load pretrained BERT tokenizer (bert-base-uncased) with dynamic padding (collator-based).
        7) Tokenize all subsets (train/val/test) with truncation to max_length=64.
        8) Initialize BERT model for sequence classification with num_labels=3.
        9) Scenario A: Fine-tune only the classifier (frozen encoder).
        10) Scenario B: Unfreeze the last encoder layer + classifier for deeper adaptation.
        11) Define TrainingArguments (batch size, learning rate, early stopping, checkpoint saving).
        12) Train both scenarios using the Hugging Face Trainer API (GPU acceleration in Colab).
        13) Evaluate best checkpoint on validation/test sets and collect performance metrics.
        14) Plot and export training/validation loss curves (plot_losses_to_pdf).
        15) Generate confusion matrices (plot_confusion_matrices) for qualitative error analysis.

    Notes:
        - Designed for execution on Google Colab GPU (with Drive integration and !pip commands).
        - The google.colab imports and shell-style pip installs (!pip ...) are Colab-specific.
        - When run locally, these lines will raise syntax or import errors (expected behavior).
        - Console outputs remain in Greek (for consistency with the original notebook).
        - Comments and docstrings are written in English for clarity in portfolio presentation.
        - The notebook version (.ipynb) includes embedded result images (RoBERTa/BERTweet).

    Maintenance:
        - For new experiments (e.g., alternative models, different unfreeze depths, or datasets),
        reuse and extend helper functions in utils.py.
        - Keep main.py focused on workflow orchestration — avoid defining utilities here.
"""

# --- NOTE ---
# This script was authored and executed in Google Colab.
# It uses google.colab.drive and google.colab.files for I/O.
# Running locally may require adapting those parts.

# --- Colab package installs  ---
!pip -q install --upgrade pip
!pip -q install "torch>=2.2" "transformers>=4.44.0" "datasets>=2.20.0" "scikit-learn>=1.3" matplotlib pandas numpy

# --- Basic libraries ---
import os
import numpy as np
import pandas as pd

# --- PyTorch ---
import torch
from datasets import Dataset

# --- Scikit-learn ---
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    classification_report
)

# --- Transformers ---
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback, DataCollatorWithPadding
)

# --- Colab I/O ---
from google.colab import drive, files

# --- pandas display options ---
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)

# --- Project utilities ---
from utils import (
    load_data,
    inspect_data,
    tokenize_text,
    plot_losses_to_pdf,
    plot_confusion_matrices
)

# ---------------------------
# Connect Google Drive in Colab
# ---------------------------
drive.mount('/content/drive')

# ---------------------------
# Create working directory in Google Drive
# ---------------------------
# Create project folder in Drive
work_dir = "/content/drive/MyDrive/bert_tweets_project"
os.makedirs(work_dir, exist_ok=True)

# Change current directory there
os.chdir(work_dir)

print("Working directory:", os.getcwd())

# ---------------------------
# Upload & Load Data
# ---------------------------
# Upload Tweets.csv from local machine to Colab environment
uploaded = files.upload()

# Load uploaded CSV into DataFrame via helper
df, _ = load_data(source='file', filepath='Tweets.csv', dataset_func=None, sheet_name=None)

# Keep only required columns (text, label) and drop rows with missing values
df = df[["text", "airline_sentiment"]].dropna()

# Initial DataFrame inspection (shape, dtypes, samples/missing values)
inspect_data(df, target_column=None)

# Remove duplicate records so each tweet appears only once
df = df.drop_duplicates()

# Show new shape (rows, columns) after deduplication
print("Νέο σχήμα DataFrame:", df.shape)

# ---------------------------
# Encode labels
# ---------------------------
# Encode 'airline_sentiment' into integer ids with LabelEncoder
le = LabelEncoder()
df["label"] = le.fit_transform(df["airline_sentiment"])

print("Classes (με τη σειρά των ids):", list(le.classes_))
print("Αριθμός κλάσεων:", len(le.classes_))

# ---------------------------
# Train/Val/Test split (80/10/10 with stratify)
# ---------------------------
# First split: 80% train, 20% temp (stratified by target to preserve class ratios)
train_X, temp_X, train_y, temp_y = train_test_split(
    df["text"], df["label"],
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

# Second split of temp: 50% validation, 50% test → overall 10% val, 10% test
val_X, test_X, val_y, test_y = train_test_split(
    temp_X, temp_y,
    test_size=0.50,
    random_state=42,
    stratify=temp_y
)

# Check shapes of the splits
print("Train:", train_X.shape, train_y.shape)
print("Val:  ", val_X.shape,   val_y.shape)
print("Test: ", test_X.shape,  test_y.shape)

# ---------------------------
# BERT Tokenizer (bert-base-uncased)
# ---------------------------
bert_Tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased",
    do_basic_tokenize=True
)

# ---------------------------
# Tokenization for train/validation/test
# ---------------------------
bert_train = tokenize_text(train_X, bert_Tokenizer, max_length=64)
bert_val   = tokenize_text(val_X,   bert_Tokenizer, max_length=64)
bert_test  = tokenize_text(test_X,  bert_Tokenizer, max_length=64)

# ---------------------------
# Load BERT + simple neural classifier (3 classes)
# ---------------------------
model_name = "bert-base-uncased"
bert_model_un1 = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Use GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model_un1.to(device)

# ---------------------------
# Freeze model — train only the classifier
# ---------------------------
# Unfreeze only the classification head
trainable_parameters = ['classifier']
for name, param in bert_model_un1.named_parameters():
    param.requires_grad = any(name.startswith(prefix) for prefix in trainable_parameters)

print("The unfreezed layers in the transformer model")
for name, param in bert_model_un1.named_parameters():
    if param.requires_grad:
        print(name)
print("--------------------------------------------------------------")
# Print the names of NOT trainable parameters
print("The freezed layers in the transformer model")
for name, param in bert_model_un1.named_parameters():
    if not param.requires_grad:
        print(name)

# ---------------------------
# Build HuggingFace Datasets (train / val / test)
# ---------------------------
train_ds = Dataset.from_dict({
    "input_ids": bert_train["input_ids"],
    "attention_mask": bert_train["attention_mask"],
    "labels": train_y.to_numpy()
})

val_ds = Dataset.from_dict({
    "input_ids": bert_val["input_ids"],
    "attention_mask": bert_val["attention_mask"],
    "labels": val_y.to_numpy()
})

test_ds = Dataset.from_dict({
    "input_ids": bert_test["input_ids"],
    "attention_mask": bert_test["attention_mask"],
    "labels": test_y.to_numpy()
})

# ---------------------------
# Data collator (dynamic padding)
# ---------------------------
data_collator = DataCollatorWithPadding(tokenizer=bert_Tokenizer, padding="longest")

print("Train size:", len(train_ds))
print("Validation size:", len(val_ds))
print("Test size:", len(test_ds))

# ---------------------------
# Output folder (checkpoints / logs)
# ---------------------------
OUTPUT_DIR = "bert_un1"

# ---------------------------
# TrainingArguments — scenario: "train only the classifier"
# ---------------------------
training_args_un1 = TrainingArguments(
    output_dir=OUTPUT_DIR,                # checkpoints/logs/metrics folder
    num_train_epochs=100,                 # many epochs: small improvements only on classifier
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=5e-5,                   # slightly larger LR for faster classifier convergence
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
    seed=42,
    fp16=torch.cuda.is_available(),
    overwrite_output_dir=True
)

# ---------------------------
# Create Trainer (scenario un1)
# ---------------------------
trainer_un1 = Trainer(
    model=bert_model_un1,                 # BERT + simple classifier (3 classes)
    args=training_args_un1,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=bert_Tokenizer,
    data_collator=data_collator,          # dynamic padding per batch
    callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]
)

# ---------------------------
# Training start
# ---------------------------
trainer_un1.train()

# ---------------------------
# Inspect best checkpoint
# ---------------------------
print("Best metric:", trainer_un1.state.best_metric)
print("Best checkpoint:", trainer_un1.state.best_model_checkpoint)

# ---------------------------
# Evaluate on test & collect loss curves
# ---------------------------
test_metrics = trainer_un1.evaluate(test_ds)
print("Test metrics:", test_metrics)

# Get logs from Trainer
logs = trainer_un1.state.log_history

# Filter entries with eval_loss
train_losses = [entry["loss"] for entry in logs if "loss" in entry and "epoch" in entry]
val_losses   = [entry["eval_loss"] for entry in logs if "eval_loss" in entry]

# Align lengths: keep the number of trained epochs
val_losses = val_losses[:len(train_losses)]

print("Train losses:", train_losses)
print("Val losses:", val_losses)

# ---------------------------
# Plot and save train/val loss curves to PDF
# ---------------------------
plot_losses_to_pdf(train_losses, val_losses, pdf_filename="loss_curves.pdf", show_plot=True)

val_metrics = trainer_un1.evaluate(eval_dataset=val_ds)
test_metrics = trainer_un1.evaluate(eval_dataset=test_ds)
print("Validation:", val_metrics)
print("Test:", test_metrics)

# ---------------------------
# Predictions on test and metrics
# ---------------------------
pred_out_un1 = trainer_un1.predict(test_ds)
y_pred_un1 = np.argmax(pred_out_un1.predictions, axis=-1)
y_true = np.array(test_y)

acc = accuracy_score(y_true, y_pred_un1)
pr_w, rc_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred_un1, average="weighted", zero_division=0)
pr_m, rc_m, f1_m, _ = precision_recall_fscore_support(y_true, y_pred_un1, average="macro", zero_division=0)

print(f"Accuracy: {acc:.4f}")
print(f"Weighted  P/R/F1: {pr_w:.4f} / {rc_w:.4f} / {f1_w:.4f}")
print(f"Macro     P/R/F1: {pr_m:.4f} / {rc_m:.4f} / {f1_m:.4f}")

# Detailed report per class
print(classification_report(y_true, y_pred_un1, target_names=list(le.classes_), zero_division=0))

# ---------------------------
# Confusion Matrices (counts + normalized) for BERT un1 (test)
# ---------------------------
plot_confusion_matrices(y_true, y_pred_un1, class_names=list(le.classes_), title_prefix="BERT un1 (test)")

# ======================================================================
# Scenario: Unfreeze only the last encoder layer (plus the classifier)
# ======================================================================

# Fresh initialization of BERT for the new scenario
bert_model_un2 = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
bert_model_un2.to(device)

# Trainable: classifier + last encoder layer only
trainable_parameters = ['classifier', 'bert.encoder.layer.11']
for name, param in bert_model_un2.named_parameters():
    param.requires_grad = any(name.startswith(prefix) for prefix in trainable_parameters)

# Output dir for scenario un2
OUTPUT_DIR = "bert_un2"

# TrainingArguments for un2: smaller LR because an encoder layer is unfrozen
training_args_un2 = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=100,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=1e-5,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
    seed=42,
    fp16=torch.cuda.is_available(),
    overwrite_output_dir=True
)

# Create Trainer — scenario un2
trainer_un2 = Trainer(
    model=bert_model_un2,               # BERT with classifier + last encoder layer unfrozen
    args=training_args_un2,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=bert_Tokenizer,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]
)

# Train
trainer_un2.train()

# Check best checkpoint
print("Best metric:", trainer_un2.state.best_metric)
print("Best checkpoint:", trainer_un2.state.best_model_checkpoint)

# Evaluate on test
test_metrics = trainer_un2.evaluate(test_ds)
print("Test metrics:", test_metrics)

# Collect logs and plot curves
logs = trainer_un2.state.log_history
train_losses = [entry["loss"] for entry in logs if "loss" in entry and "epoch" in entry]
val_losses   = [entry["eval_loss"] for entry in logs if "eval_loss" in entry]
val_losses = val_losses[:len(train_losses)]

print("Train losses:", train_losses)
print("Val losses:", val_losses)

# Plot and save curves
plot_losses_to_pdf(train_losses, val_losses, pdf_filename="loss_curves.pdf", show_plot=True)

val_metrics = trainer_un1.evaluate(eval_dataset=val_ds)
test_metrics = trainer_un1.evaluate(eval_dataset=test_ds)
print("Validation:", val_metrics)
print("Test:", test_metrics)

# Predictions and detailed metrics for un2
pred_out_un2 = trainer_un2.predict(test_ds)
y_pred_un2 = np.argmax(pred_out_un2.predictions, axis=-1)
y_true = np.array(test_y)

acc = accuracy_score(y_true, y_pred_un2)
pr_w, rc_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred_un2, average="weighted", zero_division=0)
pr_m, rc_m, f1_m, _ = precision_recall_fscore_support(y_true, y_pred_un2, average="macro", zero_division=0)

print(f"Accuracy: {acc:.4f}")
print(f"Weighted  P/R/F1: {pr_w:.4f} / {rc_w:.4f} / {f1_w:.4f}")
print(f"Macro     P/R/F1: {pr_m:.4f} / {rc_m:.4f} / {f1_m:.4f}")

# Detailed report per class
print(classification_report(y_true, y_pred_un2, target_names=list(le.classes_), zero_division=0))

# Confusion Matrices for un2 (optional step if needed)
plot_confusion_matrices(y_true, y_pred_un2, class_names=list(le.classes_), title_prefix="BERT un2 (test)")