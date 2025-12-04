# IMDB Sentiment Analysis (TF-IDF + Logistic Regression)

This project implements a Machine Learning model to classify IMDB movie reviews as **positive** or **negative**, utilizing **TF-IDF** for text vectorization and **Logistic Regression** for classification.

## 📌 Features

*   Automatic loading and processing of the IMDB dataset.
*   Text vectorization using TF-IDF.
*   Training and optimization of a Logistic Regression classifier with GridSearchCV.
*   Comprehensive model evaluation (Accuracy, Precision, Recall, F1-score) and confusion matrix generation.
*   Saving of the trained model and evaluation reports.

## 📂 Project Structure

```
analisis-sentimientos/
├── src/
│   └── train_sentiment_model.py    # Main script for training
├── models/                         # Trained models are saved here
├── reports/                        # Evaluation reports and confusion matrices
├── requirements.txt                # Project dependencies
└── README.md
```

## ▶️ How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/analisis-sentimientos.git
    cd analisis-sentimientos
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Train the model:**
    ```bash
    python src/train_sentiment_model.py
    ```

## 📦 Key Dependencies

`datasets`, `scikit-learn`, `matplotlib`, `joblib`

## 👥 Authors

- Matías Valenzuela
- Catalina Herrera