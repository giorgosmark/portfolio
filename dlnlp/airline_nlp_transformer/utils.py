"""
    utils.py — Helper functions for the Transformer-based tweet sentiment classification project (BERT focus)

    Purpose
        - Centralizes all reusable utilities for data loading, inspection, tokenization,
        evaluation metrics, and visualization (loss curves, confusion matrices).
        - Keeps main.py focused on execution logic: data paths, training loops, and experiment setup.

    Contents (indicative)
        - Data I/O & inspection:
        load_data, inspect_data
        - Tokenization:
        tokenize_text (wrapper for Hugging Face tokenizers)
        - Evaluation & visualization:
        get_preds (compute predictions and metrics)
        plot_confusion_matrices (matplotlib-based visualization)
        plot_losses_to_pdf (exports loss curves to PDF using PdfPages)

    Usage
        from utils import (
        load_data, inspect_data,
        tokenize_text, get_preds,
        plot_confusion_matrices, plot_losses_to_pdf
        )

    Dependencies
        - Core: numpy, pandas, matplotlib
        - NLP: transformers (Hugging Face)
        - ML: torch (PyTorch), scikit-learn
        - Visualization: matplotlib.backends.backend_pdf.PdfPages

    Notes
        - This module DEFINES functions only — no workflow is executed here.
        - Dataset paths, model setup, and training logic belong in main.py.
        - Functions are modular and dataset-agnostic, enabling reuse across Transformer-based NLP projects.
        - Console outputs and prints are in Greek for consistency with the notebook flow.

    Maintenance
        - When adding new tokenization, evaluation, or visualization routines,
        define them here and import explicitly in main.py.
        - Keep implementations efficient, documented, and consistent with the existing structure.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from matplotlib.backends.backend_pdf import PdfPages


def load_data(source='file', filepath=None, dataset_func=None, sheet_name=None):
    """
    Load data from a file (CSV/Excel) or a built-in sklearn dataset.

    Parameters
    ----------
    source : str, optional
        'file' for a local file (default) or 'sklearn' for an sklearn dataset.
    filepath : str or None
        File path when source='file'. Supports .csv, .xls, .xlsx.
    dataset_func : callable or None
        Function from sklearn.datasets (e.g., load_iris) when source='sklearn'.
    sheet_name : str or None
        Excel sheet name (required only for .xls/.xlsx).

    Returns
    -------
    tuple
        (df, target) where:
        df : pd.DataFrame with the data
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
    - Dtypes and count of NaN
    - Head (first rows)
    - Descriptive statistics (describe)
    - If target_column is provided: distribution of values (counts and percentages)

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to analyze.
    target_column : str, optional
        Target column (e.g., for classification). If provided, its distribution is printed.
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


def plot_confusion_matrices(y_true, y_pred, class_names, title_prefix="Model"):
    """
    Plot confusion matrices side by side:
    - Counts (absolute numbers)
    - Normalized (row-wise percentages)

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Predicted labels.
    class_names : list of str
        Class names to display on axes.
    title_prefix : str, optional
        Prefix for plot titles (default: "Model").

    Returns
    -------
    None
        Displays the confusion matrices with matplotlib.
    """
    labels = list(range(len(class_names)))

    # Compute confusion matrices
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    # Side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # 1) Counts
    im1 = axes[0].imshow(cm, cmap="Blues")
    axes[0].set_xticks(labels); axes[0].set_yticks(labels)
    axes[0].set_xticklabels(class_names); axes[0].set_yticklabels(class_names)
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")
    axes[0].set_title(f"{title_prefix} (counts)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[0].text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im1, ax=axes[0])

    # 2) Normalized
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


def tokenize_text(data, tokenizer, max_length=64):
    """
    Tokenize a collection of texts using a Hugging Face tokenizer,
    with dynamic padding applied by the data collator.

    Parameters
    ----------
    data : iterable
        Collection of texts to tokenize.
    tokenizer : transformers.PreTrainedTokenizer
        Hugging Face tokenizer to use.
    max_length : int, optional, default=64
        Maximum tokenized sequence length.
        Longer texts are truncated.
        Padding is NOT applied here, but dynamically by the collator.

    Returns
    -------
    dict
        A dictionary containing tokenized data (e.g., input_ids, attention_mask).
    """
    # Apply truncation to max_length tokens
    # DO NOT pad here -> dynamic padding in the collator
    # The collator matches padding to the longest sequence in the batch
    # This reduces unnecessary padding and speeds up training
    return tokenizer(
        list(data),
        truncation=True,
        max_length=max_length,
        padding=False,   # dynamic padding in the collator
    )


def plot_losses_to_pdf(
        train_losses,
        val_losses,
        pdf_filename='training_validation_losses.pdf',
        figsize=(8, 6),
        show_plot=False,
):
    """
    Plot training and validation losses,
    and save the figure to a PDF file.

    Args
    ----
    train_losses (list): Training loss per epoch.
    val_losses (list): Validation loss per epoch.
    pdf_filename (str): Name/path of the PDF file to save.
    figsize (tuple): Figure size (width, height in inches).
    show_plot (bool): Whether to display the figure.

    Return
    ------
    pdf_filename (str): The filename of the generated PDF.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)

    # Plot the train/val curves
    ax.plot(train_losses, label="Train Loss")
    ax.plot(val_losses, label="Val Loss")

    ax.set_title("Training & Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()

    # Save the current figure into a PDF
    with PdfPages(pdf_filename) as pdf:
        pdf.savefig(fig)

    if show_plot:
        plt.show()

    plt.close(fig)
    print(f"The learning curves are generated in pdf: {pdf_filename}")
    return pdf_filename