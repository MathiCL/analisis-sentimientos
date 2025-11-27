# Sentiment Analysis IMDB (TF-IDF + Logistic Regression)

Este proyecto entrena un modelo de Machine Learning para clasificar reseñas de películas del dataset **IMDB** en dos categorías: **positivo** y **negativo**, utilizando técnicas de NLP como TF-IDF y un clasificador Logistic Regression.

---

## 📌 Funcionalidad
- Carga automática del dataset IMDB desde `datasets`.
- Limpieza y vectorización del texto utilizando **TF-IDF**.
- Entrenamiento de un modelo **Logistic Regression**.
- Evaluación con métricas: Accuracy, Precision, Recall, F1-score.
- Matrices de confusión generadas automáticamente.
- GridSearchCV para mejorar el modelo.
- Guardado del modelo final en `/models`.

---

## 📂 Estructura del proyecto

analisis-sentimientos/
│
├── src/
│ └── train_sentiment_model.py
├── models/
├── reports/
├── requirements.txt
└── README.md

---

## ▶️ Cómo ejecutar

1. **Clonar el repositorio**
```bash
git clone https://github.com/TU_USUARIO/analisis-sentimientos.git
cd analisis-sentimientos

2. Crear entorno virtual (opcional)

python -m venv venv
venv\Scripts\activate

3. Instalar dependencias

pip install -r requirements.txt

4. Ejecutar el script

python src/train_sentiment_model.py

📦 Dependencias principales:

datasets
scikit-learn
matplotlib
joblib

👥 Autores

-Matías Valenzuela

-Catalina Herrera 

