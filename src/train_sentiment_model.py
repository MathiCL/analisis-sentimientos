"""
Train a sentiment analysis model on the IMDB dataset using
TF-IDF + Logistic Regression.

- Loads IMDB dataset from HuggingFace `datasets`
- Splits train into train/validation
- Vectorizes text with TF-IDF
- Trains a Logistic Regression classifier
- Runs GridSearchCV for hyperparameter tuning
- Evaluates on validation and test sets
- Saves confusion matrix plot
- Saves the trained model to disk
- Provides a helper function to predict on custom texts
"""

import os
from typing import List, Tuple

from datasets import load_dataset

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

import matplotlib.pyplot as plt
import numpy as np
import joblib


# ----------------------------- Config ----------------------------- #

RANDOM_STATE = 42
MAX_FEATURES = 20000          # número máximo de tokens en el vocabulario
NGRAM_RANGE = (1, 2)          # unigrams + bigrams
MODEL_DIR = os.path.join("models")
REPORTS_DIR = os.path.join("reports")


# --------------------------- Utilidades --------------------------- #

def ensure_directories():
    """Create folders for models and reports if they don't exist."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)


def load_imdb_dataset():
    """
    Load IMDB dataset using HuggingFace `datasets`.
    Returns:
        train_texts, train_labels, test_texts, test_labels
    """
    print("Loading IMDB dataset from HuggingFace...")
    dataset = load_dataset("imdb")

    train_texts = dataset["train"]["text"]
    train_labels = dataset["train"]["label"]

    test_texts = dataset["test"]["text"]
    test_labels = dataset["test"]["label"]

    print(f"Train size: {len(train_texts)}")
    print(f"Test size : {len(test_texts)}")
    print()
    return train_texts, train_labels, test_texts, test_labels


def create_train_val_split(
    texts, labels, val_size: float = 0.2
) -> Tuple[List[str], List[str], List[int], List[int]]:
    """
    Split training data into train and validation subsets.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        texts,
        labels,
        test_size=val_size,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    print(f"Train subset size: {len(X_train)}")
    print(f"Validation size  : {len(X_val)}")
    print()
    return X_train, X_val, y_train, y_val


def build_pipeline() -> Pipeline:
    """
    Build a scikit-learn Pipeline: TF-IDF vectorizer + Logistic Regression.
    """
    tfidf = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        stop_words="english",
    )

    clf = LogisticRegression(
        max_iter=200,
        n_jobs=-1,
    )

    pipe = Pipeline(
        steps=[
            ("tfidf", tfidf),
            ("clf", clf),
        ]
    )

    return pipe


def evaluate_model(
    name: str, model: Pipeline, X, y, save_confusion: bool = False, split_name: str = ""
):
    """
    Print standard classification metrics and optionally save confusion matrix.
    """
    y_pred = model.predict(X)

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    print(f"===== {name} =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print()
    print("Classification report:")
    print(classification_report(y, y_pred, target_names=["negative", "positive"]))
    print("-" * 60)

    if save_confusion:
        cm = confusion_matrix(y, y_pred)
        plot_confusion_matrix(
            cm,
            classes=["negative", "positive"],
            title=f"Confusion matrix ({split_name})",
            filename=os.path.join(REPORTS_DIR, f"confusion_matrix_{split_name}.png"),
        )


def plot_confusion_matrix(cm, classes, title: str, filename: str):
    """
    Create and save confusion matrix plot.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix to: {filename}")


def run_grid_search(pipe: Pipeline, X_train, y_train) -> GridSearchCV:
    """
    Run GridSearchCV over a small hyperparameter grid.
    """
    print("Running GridSearchCV (this may take some minutes)...")

    param_grid = {
        "tfidf__max_features": [10000, 20000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "clf__C": [0.5, 1.0, 2.0],
    }

    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=3,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
    )

    grid.fit(X_train, y_train)

    print("\n===== GridSearchCV results =====")
    print("Best params:", grid.best_params_)
    print(f"Best CV F1: {grid.best_score_:.4f}")
    print("-" * 60)
    print()

    return grid


def save_model(model: Pipeline, filename: str):
    """
    Save trained pipeline to disk using joblib.
    """
    joblib.dump(model, filename)
    print(f"Saved trained model to: {filename}")


def load_model(filename: str) -> Pipeline:
    """
    Load a trained pipeline from disk.
    """
    return joblib.load(filename)


def predict_custom_texts(model: Pipeline, texts: List[str]):
    """
    Print predictions (positive/negative) for a list of custom texts.
    """
    preds = model.predict(texts)
    for text, label in zip(texts, preds):
        sentiment = "positive" if label == 1 else "negative"
        print(f'"{text}" -> {sentiment}')


# ------------------------------ main ------------------------------ #

def main():
    ensure_directories()

    # 1. Load dataset
    train_texts, train_labels, test_texts, test_labels = load_imdb_dataset()

    # 2. Create train/validation split
    X_train, X_val, y_train, y_val = create_train_val_split(train_texts, train_labels)

    # 3. Build baseline pipeline
    base_pipe = build_pipeline()

    print("Training baseline model...")
    base_pipe.fit(X_train, y_train)
    print("Done.\n")

    # 4. Evaluate baseline on validation set
    evaluate_model(
        "Baseline (validation)", base_pipe, X_val, y_val, split_name="val_baseline"
    )

    # 5. Grid search for better hyperparameters (on training data)
    grid = run_grid_search(build_pipeline(), X_train, y_train)
    best_model = grid.best_estimator_

    # 6. Evaluate best model on validation set
    evaluate_model(
        "Best model after GridSearch (validation)",
        best_model,
        X_val,
        y_val,
        save_confusion=True,
        split_name="val_best",
    )

    # 7. Evaluate best model on test set (final metric)
    print("\n=== Final evaluation on TEST set ===")
    evaluate_model(
        "Best model (test)",
        best_model,
        test_texts,
        test_labels,
        save_confusion=True,
        split_name="test_best",
    )

    # 8. Save trained model
    model_path = os.path.join(MODEL_DIR, "imdb_sentiment_logreg.joblib")
    save_model(best_model, model_path)

    # 9. Try on some custom examples
    print("\nSome custom predictions:")
    sample_texts = [
        "This movie was amazing, I loved every second of it!",
        "The film was boring and too long. I would not recommend it.",
        "It was okay, not the best but not terrible either.",
    ]
    predict_custom_texts(best_model, sample_texts)


if __name__ == "__main__":
    main()
