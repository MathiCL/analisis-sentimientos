import numpy as np

def sigmoid(z):
    """
    Función de activación sigmoide, utilizada en la regresión logística para
    mapear cualquier valor real al rango (0, 1).
    Esta probabilidad puede interpretarse como la probabilidad de que la
    muestra pertenezca a la clase positiva.

    Args:
        z (float or np.ndarray): La entrada lineal (producto punto de pesos y
        características).

    Returns:
        float or np.ndarray: El valor de la función sigmoide, entre 0 y 1.
    """
    return 1 / (1 + np.exp(-z))


class LogisticRegression:
    """
    Implementación de regresión logística desde cero para clasificación binaria.
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000, C=1.0):
        """
        Inicializa el clasificador de Regresión Logística.

        Args:
            learning_rate (float): La tasa de aprendizaje para el descenso de
            gradiente.
            n_iterations (int): El número de iteraciones para el entrenamiento.
            C (float): Parámetro de regularización. Un valor más pequeño
            especifica una regularización más fuerte.
                       No se usa directamente en el descenso de gradiente
                       simple, pero es un hiperparámetro común.
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.C = C  # Guardado para posible uso futuro, como en la función
                    # de coste con regularización.
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Entrena el modelo de regresión logística utilizando el descenso de
        gradiente.

        Args:
            X (np.ndarray or scipy.sparse.csr_matrix): Las características
            de entrenamiento.
            y (np.ndarray): Las etiquetas de destino (0 o 1).
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Convert y to numpy array if it's not
        y = np.asarray(y)

        # Descenso de gradiente
        for _ in range(self.n_iterations):
            linear_model = X.dot(self.weights) + self.bias
            y_predicted = sigmoid(linear_model)

            # Cálculo de los gradientes
            dw = (1 / n_samples) * X.T.dot(y_predicted - y)
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Actualización de los pesos y el sesgo
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X):
        """
        Calcula las probabilidades de clase para las muestras de entrada.

        Args:
            X (np.ndarray or scipy.sparse.csr_matrix): Las características de
            entrada.

        Returns:
            np.ndarray: Un array con las probabilidades para la clase
            positiva (clase 1).
        """
        linear_model = X.dot(self.weights) + self.bias
        return sigmoid(linear_model)

    def predict(self, X):
        """
        Predice las etiquetas de clase para las muestras de entrada.

        Args:
            X (np.ndarray or scipy.sparse.csr_matrix): Las características de
            entrada.

        Returns:
            np.ndarray: Las etiquetas de clase predichas (0 o 1).
        """
        y_predicted_proba = self.predict_proba(X)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted_proba]
        return np.array(y_predicted_cls)

    def get_params(self, deep=True):
        """
        Devuelve los parámetros del estimador, útil para la compatibilidad
        con scikit-learn.
        """
        return {"learning_rate": self.learning_rate, "n_iterations": self.n_iterations, "C": self.C}

    def set_params(self, **params):
        """
        Establece los parámetros del estimador.
        """
        for param, value in params.items():
            setattr(self, param, value)
        return self
