"""
    main.py — Main workflow for the text classification (sentiment analysis) project

    Purpose:
        - Executes the full supervised NLP pipeline for airline tweet sentiment analysis.
        - Handles dataset loading, text preprocessing, feature extraction (TF-IDF & Word2Vec),
          model training (ANN), evaluation, and comparison of both vectorization methods.
        - Keeps reusable functions in utils.py to maintain a clean and modular structure.

    Workflow Overview:
        1. Load the dataset (Tweets.csv) and perform initial inspection (info, NaNs, head, stats).
        2. Select relevant columns (text, airline_sentiment) and label-encode sentiment classes.
        3. Clean raw text (remove HTML, mentions, URLs, noise, normalize casing/spaces).
        4. Tokenize and lemmatize text using spaCy, removing stopwords and compressing repetitions.
        5. Explore most frequent n-grams (uni/bigrams) across all texts.
        6. Remove empty or invalid rows (e.g., missing or empty token lists).
        7. Vectorize texts with:
              a) TF-IDF (unigram + bigram features)
              b) Word2Vec (average pre-trained word embeddings)
        8. Split each representation into train/validation/test sets (80/10/10 stratified).
        9. Build and train a simple feedforward ANN for each representation:
              - 2 hidden layers, ReLU activations, Dropout, Early Stopping, LR scheduler.
       10. Monitor and export training/validation loss curves to PDF.
       11. Evaluate models on the test set and print classification reports.
       12. Plot confusion matrices (raw counts and normalized) for both TF-IDF and Word2Vec.
       13. Compare overall accuracy and performance metrics between the two vectorization methods.

    Notes:
        - All reusable utilities (preprocessing, model, training, plotting) are defined in utils.py.
        - Adjust dataset path in the load_data() call as needed (e.g., datasets/Tweets.csv).
        - Update the Word2Vec model path (variable `w2v_path`) to match your local filesystem.
          Example: w2v_path = r"C:\\Users\\<username>\\Projects\\word2vec-google-news-300.model"
        - The script runs entirely on CPU by default (no GPU required).
        - pandas display options and matplotlib visualizations are configured locally for convenience.

    Maintenance:
        - When adding new experiments (e.g., LSTM, BERT, or different embeddings),
          keep preprocessing and vectorization logic in utils.py.
        - This script should remain focused on high-level workflow orchestration only.
"""

# --- Basic libraries ---
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader

# --- Scikit-learn ---
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# --- Project utilities ---
from utils import (
    load_data,
    inspect_data,
    clean_text,
    spacy_preprocess_batch,
    top_ngrams,
    plot_bar,
    vectorize_texts,
    ANN,
    train_model,
    plot_losses_to_pdf,
    get_preds,
)

# --- pandas display settings ---
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)


# -----------------------------
# Load dataset (Tweets.csv)
# -----------------------------
df, _ = load_data(source='file', filepath='datasets/Tweets.csv', dataset_func=None, sheet_name=None)

# -----------------------------
# Inspect dataset
# -----------------------------
inspect_data(df, target_column=None)

# -----------------------------
# Keep only useful columns
# -----------------------------
df = df[["text", "airline_sentiment"]].dropna()

# -----------------------------
# Label Encoding
# -----------------------------
le = LabelEncoder()
df["label"] = le.fit_transform(df["airline_sentiment"])

print("Classes (με τη σειρά των ids):", list(le.classes_))
print("Αριθμός κλάσεων:", len(le.classes_))

print("\nΚατανομή κλάσεων (counts):")
print(df["airline_sentiment"].value_counts())

print("\nΚατανομή κλάσεων (ποσοστά):")
print(df["airline_sentiment"].value_counts(normalize=True) * 100)

# Optional head preview (script-friendly)
print(df["text"].head(5))

# -----------------------------
# Text cleaning
# -----------------------------
df.loc[:, "cleaned_text"] = df["text"].apply(clean_text)
print(df["cleaned_text"].head(5))

# -----------------------------
# spaCy tokenization & lemmatization
# -----------------------------
tokens_col, lemmas_col = spacy_preprocess_batch(df["cleaned_text"].tolist(), batch_size=800)
df["tokens"] = tokens_col
df["lemmas_str"] = lemmas_col
print(df[["tokens"]].head(5))

# -----------------------------
# n-gram exploration (overall)
# -----------------------------
uni_overall = top_ngrams(df["lemmas_str"], (1, 1), top_k=20, min_df=5)
bi_overall = top_ngrams(df["lemmas_str"], (2, 2), top_k=20, min_df=5)

plot_bar(uni_overall, "Top-20 Unigrams (lemmas, overall, min_df=5)")
plot_bar(bi_overall, "Top-20 Bigrams (lemmas, overall, min_df=5)")

# -----------------------------
# Remove empty/NaN after spaCy
# -----------------------------
mask_tokens_nan = df["tokens"].isna()
mask_tokens_empty = df["tokens"].map(lambda x: len(x) == 0)

mask_lemmas_nan = df["lemmas_str"].isna()
mask_lemmas_empty = df["lemmas_str"].astype(str).str.strip().eq("")

print("NaN tokens:", mask_tokens_nan.sum())
print("Empty tokens:", mask_tokens_empty.sum())
print("NaN lemmas_str:", mask_lemmas_nan.sum())
print("Empty lemmas_str:", mask_lemmas_empty.sum())

bad_mask = mask_tokens_nan | mask_tokens_empty | mask_lemmas_nan | mask_lemmas_empty
print("\nΣύνολο που θα αφαιρεθούν:", bad_mask.sum())

df = df.loc[~bad_mask].reset_index(drop=True)
print("Νέο μέγεθος df:", len(df))

# Double-check
mask_tokens_nan = df["tokens"].isna()
mask_tokens_empty = df["tokens"].map(lambda x: len(x) == 0)

mask_lemmas_nan = df["lemmas_str"].isna()
mask_lemmas_empty = df["lemmas_str"].astype(str).str.strip().eq("")

print("NaN tokens:", mask_tokens_nan.sum())
print("Empty tokens:", mask_tokens_empty.sum())
print("NaN lemmas_str:", mask_lemmas_nan.sum())
print("Empty lemmas_str:", mask_lemmas_empty.sum())

# -----------------------------
# TF-IDF Vectorization
# -----------------------------
labels = df["label"].values
vector_method = "tfidf"
X, tfidf_vec = vectorize_texts(df, "lemmas_str", method=vector_method, max_features=15000)
print("Shape of X:", X.shape)

# -----------------------------
# Train/Validation/Test split
# -----------------------------
train_X, temp_X, train_y, temp_y = train_test_split(
    X, labels, test_size=0.20, random_state=42, stratify=labels
)

val_X, test_X, val_y, test_y = train_test_split(
    temp_X, temp_y, test_size=0.50, random_state=42, stratify=temp_y
)
del temp_X, temp_y

# -----------------------------
# Convert to PyTorch tensors
# -----------------------------
train_X_tensor = torch.tensor(train_X, dtype=torch.float32)
train_y_tensor = torch.tensor(train_y, dtype=torch.long)

val_X_tensor = torch.tensor(val_X, dtype=torch.float32)
val_y_tensor = torch.tensor(val_y, dtype=torch.long)

test_X_tensor = torch.tensor(test_X, dtype=torch.float32)
test_y_tensor = torch.tensor(test_y, dtype=torch.long)

# -----------------------------
# TensorDataset & DataLoader
# -----------------------------
train_dataset = TensorDataset(train_X_tensor, train_y_tensor)
val_dataset = TensorDataset(val_X_tensor, val_y_tensor)
test_dataset = TensorDataset(test_X_tensor, test_y_tensor)

batch_size = 32
train_loader_idf = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader_idf = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader_idf = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print('The preprocessing of the data is finished and the model is ready for training')

# -----------------------------
# Model definition (TF-IDF)
# -----------------------------
input_dim = X.shape[1]
output_dim = len(le.classes_)

model_ann_idf = ANN(
    input_dim=input_dim,
    output_dim=output_dim,
    hidden_layer1=256,
    hidden_layer2=256,
    dropoutp1=0.3,
    dropoutp2=0.3
)

# -----------------------------
# Loss & optimizer
# -----------------------------
# Example for weighted classes
# classes = np.unique(train_y)
# cls_weights = compute_class_weight('balanced', classes=classes, y=train_y)
# cls_weights = torch.tensor(cls_weights, dtype=torch.float32)
# loss_function = torch.nn.CrossEntropyLoss(weight=cls_weights)

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_ann_idf.parameters(), lr=5e-4)

# -----------------------------
# Scheduler & Early stopping
# -----------------------------
num_epochs = 200
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda e: max(0.0, 1.0 - e / float(num_epochs - 1))
)
patience = 7

# -----------------------------
# Train (TF-IDF)
# -----------------------------
model_idf, train_losses_idf, val_losses_idf, train_accs_idf, val_accs_idf = train_model(
    model=model_ann_idf,
    train_loader=train_loader_idf,
    val_loader=val_loader_idf,
    criterion=loss_function,
    optimizer=optimizer,
    num_epochs=num_epochs,
    patience=patience,
    scheduler=scheduler
)

# -----------------------------
# Plot losses to PDF
# -----------------------------
plot_losses_to_pdf(train_losses=train_losses_idf,
                   val_losses=val_losses_idf,
                   pdf_filename="training_validation_losses.pdf",
                   figsize=(12, 8),
                   show_plot=True)

# -----------------------------
# Evaluate on test set (TF-IDF)
# -----------------------------
model_idf.eval()
correct, total = 0, 0
with torch.no_grad():
    for xb, yb in test_loader_idf:
        logits = model_idf(xb.float())
        preds = torch.argmax(logits, dim=1)
        correct += (preds == yb.long()).sum().item()
        total += yb.size(0)
test_acc = 100.0 * correct / total
print(f"Test Accuracy: {test_acc:.2f}%")

# Detailed report (TF-IDF)
y_true, y_pred = get_preds(model_idf, test_loader_idf)
print(classification_report(y_true, y_pred, digits=3, zero_division=0))

# -----------------------------
# Word2Vec path & model (loaded inside utils during vectorization)
# -----------------------------
w2v_path = r"C:\Users\giorg\PycharmProjects\PythonProject4\word2vec-google-news-300.model"

# -----------------------------
# Word2Vec vectorization (average embeddings)
# -----------------------------
X_w2v = vectorize_texts(
    df,
    feature_name="tokens",
    method="word2vec",
    model_path=w2v_path
)
labels_w2v = df["label"].values
print("Shape (W2V):", X_w2v.shape)

# -----------------------------
# Split (W2V)
# -----------------------------
train_X_w2v, temp_X_w2v, train_y_w2v, temp_y_w2v = train_test_split(
    X_w2v, labels_w2v, test_size=0.20, random_state=42, stratify=labels_w2v
)

val_X_w2v, test_X_w2v, val_y_w2v, test_y_w2v = train_test_split(
    temp_X_w2v, temp_y_w2v, test_size=0.50, random_state=42, stratify=temp_y_w2v
)

# -----------------------------
# Tensors (W2V)
# -----------------------------
train_X_tensor_w2v = torch.tensor(train_X_w2v, dtype=torch.float32)
train_y_tensor_w2v = torch.tensor(train_y_w2v, dtype=torch.long)

val_X_tensor_w2v = torch.tensor(val_X_w2v, dtype=torch.float32)
val_y_tensor_w2v = torch.tensor(val_y_w2v, dtype=torch.long)

test_X_tensor_w2v = torch.tensor(test_X_w2v, dtype=torch.float32)
test_y_tensor_w2v = torch.tensor(test_y_w2v, dtype=torch.long)

train_dataset_w2v = TensorDataset(train_X_tensor_w2v, train_y_tensor_w2v)
val_dataset_w2v = TensorDataset(val_X_tensor_w2v, val_y_tensor_w2v)
test_dataset_w2v = TensorDataset(test_X_tensor_w2v, test_y_tensor_w2v)

batch_size = 32
train_loader_w2v = DataLoader(train_dataset_w2v, batch_size=batch_size, shuffle=True)
val_loader_w2v = DataLoader(val_dataset_w2v, batch_size=batch_size, shuffle=False)
test_loader_w2v = DataLoader(test_dataset_w2v, batch_size=batch_size, shuffle=False)

print('The preprocessing of the W2V data is finished and the model is ready for training')

# -----------------------------
# Model (W2V)
# -----------------------------
input_dim_w2v = X_w2v.shape[1]
output_dim_w2v = len(le.classes_)

model_ann_w2v = ANN(
    input_dim=input_dim_w2v,
    output_dim=output_dim_w2v,
    hidden_layer1=256,
    hidden_layer2=256,
    dropoutp1=0.3,
    dropoutp2=0.3
)

criterion_w2v = torch.nn.CrossEntropyLoss()
optimizer_w2v = torch.optim.Adam(model_ann_w2v.parameters(), lr=5e-4)

num_epochs_w2v = 200
scheduler_w2v = torch.optim.lr_scheduler.LambdaLR(
    optimizer_w2v,
    lr_lambda=lambda e: max(0.0, 1.0 - e / float(num_epochs_w2v - 1))
)
patience_w2v = 7

# -----------------------------
# Train (W2V)
# -----------------------------
model_w2v, train_losses_w2v, val_losses_w2v, train_accs_w2v, val_accs_w2v = train_model(
    model=model_ann_w2v,
    train_loader=train_loader_w2v,
    val_loader=val_loader_w2v,
    criterion=criterion_w2v,
    optimizer=optimizer_w2v,
    num_epochs=num_epochs_w2v,
    patience=patience_w2v,
    scheduler=scheduler_w2v
)

# Plot losses (W2V) to PDF (optional reusing same filename)
plot_losses_to_pdf(train_losses=train_losses_w2v,
                   val_losses=val_losses_w2v,
                   pdf_filename="training_validation_losses.pdf",
                   figsize=(12, 8),
                   show_plot=True)

# -----------------------------
# Test (W2V)
# -----------------------------
model_w2v.eval()
correct_w2v, total_w2v = 0, 0
with torch.no_grad():
    for xb, yb in test_loader_w2v:
        preds = model_w2v(xb.float()).argmax(1)
        correct_w2v += (preds == yb).sum().item()
        total_w2v += yb.size(0)
test_acc_w2v = 100.0 * correct_w2v / total_w2v
print(f"Test Accuracy (W2V): {test_acc_w2v:.2f}%")

# Detailed report (W2V)
y_true_w2v, y_pred_w2v = get_preds(model_w2v, test_loader_w2v)
print(classification_report(y_true_w2v, y_pred_w2v, digits=3, zero_division=0))

# -----------------------------
# Confusion matrices (TF-IDF vs W2V)
# -----------------------------
y_true_idf, y_pred_idf = get_preds(model_idf, test_loader_idf)
y_true, y_pred = get_preds(model_w2v, test_loader_w2v)

labels_idx = [0, 1, 2]
class_names = ["neg", "neu", "pos"]

cm_idf = confusion_matrix(y_true_idf, y_pred_idf, labels=labels_idx)
cm_w2v = confusion_matrix(y_true, y_pred, labels=labels_idx)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
for ax, cm, title in zip(axes, [cm_idf, cm_w2v], ["IDF (counts)", "W2V (counts)"]):
    im = ax.imshow(cm)
    ax.set_xticks(labels_idx); ax.set_yticks(labels_idx)
    ax.set_xticklabels(class_names); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
plt.tight_layout(); plt.show()

# Normalized confusion matrices
cm_idf_norm = cm_idf.astype(float) / cm_idf.sum(axis=1, keepdims=True)
cm_w2v_norm = cm_w2v.astype(float) / cm_w2v.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
for ax, cm, title in zip(axes, [cm_idf_norm, cm_w2v_norm], ["TF-IDF (norm)", "W2V (norm)"]):
    im = ax.imshow(cm)
    ax.set_xticks(labels_idx); ax.set_yticks(labels_idx)
    ax.set_xticklabels(class_names); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center")
plt.tight_layout(); plt.show()