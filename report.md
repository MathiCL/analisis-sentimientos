# Report: Sentiment Analysis of IMDB Movie Reviews

**Group Members:** Matías Valenzuela, Catalina Herrera
**Date:** November 27, 2025

---

## 1. Introduction

The objective of this project is to develop and evaluate a machine learning model capable of performing sentiment analysis on movie reviews. The task is to classify a given review text as either **positive** or **negative**.

This is a classic Natural Language Processing (NLP) binary classification problem. We chose the well-known IMDB movie review dataset for this task and implemented a Logistic Regression classifier, a robust and interpretable linear model suitable for this kind of problem.

## 2. Dataset

We used the `imdb` dataset, which is publicly available from HuggingFace's `datasets` library.

-   **Source:** `datasets.load_dataset('imdb')`
-   **Total Size:** 50,000 reviews.
-   **Splits:** The dataset is pre-divided into a training set of 25,000 reviews and a testing set of 25,000 reviews.
-   **Balance:** The dataset is well-balanced, with an equal number of positive and negative reviews in both splits.

For our training process, we further subdivided the 25,000-review training set into a smaller training subset (20,000 reviews) and a validation subset (5,000 reviews). The validation set is crucial for tuning the model's hyperparameters without introducing bias from the final test set.

## 3. Methodology

Our approach follows a standard NLP machine learning pipeline: data preprocessing, feature extraction, model training, and evaluation.

### 3.1. Preprocessing and Feature Extraction

The raw text of movie reviews cannot be fed directly into a machine learning algorithm. It must be converted into a numerical representation. We used the **Term Frequency-Inverse Document Frequency (TF-IDF)** vectorization technique.

-   **TF-IDF:** This method transforms text into a meaningful numerical representation by weighting words based on their frequency in a document and their inverse frequency across the entire corpus. It effectively gives more importance to words that are frequent in a specific review but rare in other reviews.

### 3.2. Model Selection

We chose **Logistic Regression** as our classification algorithm. It's a linear model that is efficient, highly interpretable, and serves as a very strong baseline for text classification tasks. The model calculates the probability that a review belongs to the "positive" class.

### 3.3. Training and Hyperparameter Tuning

To achieve the best possible performance, we implemented a two-stage training process:

1.  **Baseline Model:** We first trained a default `LogisticRegression` model to establish a performance baseline. This helps us understand the effectiveness of hyperparameter tuning.

2.  **Hyperparameter Tuning with GridSearchCV:** We used `GridSearchCV` to systematically search for the optimal combination of hyperparameters for both the TF-IDF vectorizer and the Logistic Regression classifier. `GridSearchCV` performs a k-fold cross-validation (with k=3 in our case) for each combination of parameters to prevent overfitting and find the most generalizable model.

The hyperparameters we tuned were:
-   `tfidf__ngram_range`: The range of n-grams to consider (e.g., single words (1,1) or words and pairs of words (1,2)).
-   `tfidf__max_features`: The total number of most frequent words to include in the vocabulary.
-   `clf__C`: The inverse of regularization strength for the Logistic Regression model. A smaller `C` value specifies stronger regularization.

## 4. Results

The model was trained and its hyperparameters were tuned using the training and validation sets. The final, optimized model was then evaluated on the unseen test set to provide an unbiased assessment of its performance.

### 4.1. Hyperparameter Tuning Results

The `GridSearchCV` process identified the following optimal parameters:

-   **Best Parameters:**
    -   `C` (Regularization): `2.0`
    -   `max_features` (Vocabulary size): `20000`
    -   `ngram_range`: `(1, 2)` (meaning both unigrams and bigrams were used)

### 4.2. Final Model Performance (on Test Set)

The performance of the final, optimized model on the 25,000-review test set is the definitive measure of our success.

-   **Accuracy:** `0.8810` (or 88.10%)
-   **Precision:** `0.8808`
-   **Recall:** `0.8814`
-   **F1-Score:** `0.8811`

These metrics indicate that the model is highly effective and well-balanced. An F1-score of 0.8811 shows a strong balance between precision (the model's ability to not label a negative sample as positive) and recall (the model's ability to find all the positive samples).

### 4.3. Confusion Matrix

The confusion matrix provides a visual representation of the model's performance on the test set.

*(The confusion matrix image `reports/confusion_matrix_test_best.png` would be embedded here in the final document.)*

The matrix shows the number of True Positives, True Negatives, False Positives, and False Negatives, confirming the strong classification results.

## 5. Conclusion

We successfully implemented a machine learning pipeline to classify IMDB movie reviews with high accuracy. By using TF-IDF for feature extraction and a Logistic Regression classifier, we achieved a final F1-score of **88.11%** on the unseen test data.

The use of `GridSearchCV` was crucial for optimizing the model, leading to a noticeable improvement over the baseline. The results demonstrate that even relatively simple linear models, when properly tuned, can be extremely effective for text classification tasks. The final model is robust, accurate, and provides a reliable solution to the problem of sentiment analysis.
