# Presentation Outline: Sentiment Analysis of IMDB Reviews

---

### Slide 1: Title Slide

-   **Title:** Sentiment Analysis of IMDB Movie Reviews
-   **Subtitle:** A Machine Learning Approach
-   **Group:** Matías Valenzuela, Catalina Herrera
-   **Date:** November 27, 2025

---

### Slide 2: The Goal

-   **Objective:** Automatically classify a movie review as **Positive** or **Negative**.
-   **Why?:** This has applications in customer feedback analysis, brand monitoring, and content recommendation.
-   **Our Task:**
    -   **Task:** Binary Text Classification
    -   **Dataset:** IMDB Movie Reviews
    -   **Algorithm:** Logistic Regression

---

### Slide 3: The Dataset: IMDB Reviews

-   A collection of 50,000 movie reviews.
-   Perfectly balanced:
    -   25,000 for training (12.5k positive, 12.5k negative)
    -   25,000 for testing (12.5k positive, 12.5k negative)
-   We created an additional **validation set** (5,000 reviews) from the training data for fine-tuning our model.

---

### Slide 4: Our Methodology: The Pipeline

-   A 4-step process:
    1.  **Load Data:** Import the IMDB dataset.
    2.  **Vectorize Text:** Convert words into numbers using **TF-IDF**. This tells the model how important a word is to a review.
    3.  **Train Model:** Teach a **Logistic Regression** model to distinguish between positive and negative reviews.
    4.  **Evaluate:** Test the model on data it has never seen before.

---

### Slide 5: Step 1 - Feature Extraction with TF-IDF

-   **Problem:** Machines don't understand words, they understand numbers.
-   **Solution:** **TF-IDF** (Term Frequency-Inverse Document Frequency)
    -   **TF (Term Frequency):** How often does a word appear in a review?
    -   **IDF (Inverse Document Frequency):** How rare is that word across all reviews?
-   **Result:** Important words get a high score. Common words like "the" or "a" get a low score.

---

### Slide 6: Step 2 - Model Training & Optimization

-   **Algorithm:** **Logistic Regression**
    -   A simple, fast, and effective linear model for classification.
-   **Optimization:** How do we find the *best* version of our model?
    -   We used **GridSearchCV**.
    -   It's an automated process that tests different hyperparameter combinations to find the one with the best performance.
    -   **Key parameters tuned:** n-gram range, vocabulary size, and regularization strength (C).

---

### Slide 7: The Winning Combination

-   GridSearchCV found the best settings for our model:
    -   **N-grams:** `(1, 2)` -> The model looks at both single words ("amazing") and pairs of words ("very good"). This captures more context.
    -   **Vocabulary Size:** `20,000` features -> The model focuses on the 20,000 most important words.
    -   **Regularization (C):** `2.0` -> A good balance to prevent overfitting.

---

### Slide 8: The Results: How Well Did It Do?

-   We evaluated the final, optimized model on the **25,000 unseen test reviews**.
-   **Final Accuracy: 88.1%**
-   **Key Metrics:**
    -   **Precision:** 88.1% (When it predicts "positive", it's right 88.1% of the time).
    -   **Recall:** 88.1% (It correctly identifies 88.1% of all actual "positive" reviews).
    -   **F1-Score: 88.1%** (A combined measure of Precision and Recall, showing a robust and balanced model).

---

### Slide 9: Confusion Matrix (Visualizing the Results)

-   *(Show the `reports/confusion_matrix_test_best.png` image here)*
-   **What it shows:**
    -   High numbers on the diagonal (top-left to bottom-right) are **correct** predictions.
    -   Low numbers on the off-diagonal are **incorrect** predictions.
-   **Our result:** The model is very good at correctly identifying both positive and negative reviews, with a similar number of errors for each class.

---

### Slide 10: Conclusion

-   **Success!** We built a highly effective sentiment analysis model with an **88.1% F1-score**.
-   **Key Takeaways:**
    -   Proper data splitting (train/validation/test) is essential.
    -   TF-IDF is a powerful technique for text feature extraction.
    -   Hyperparameter tuning with GridSearchCV significantly improves model performance.
    -   Logistic Regression, while simple, is a very strong baseline for text classification.

---

### Slide 11: Q&A

-   **Thank you!**
-   **Any questions?**
