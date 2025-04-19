import os
import re
import pickle
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)
CORS(app)

# Define LogisticRegressionManual class
class LogisticRegressionManual:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def initialize_weights_and_bias(self, dimension):
        w = np.zeros((dimension, 1))
        b = 0.0
        return w, b

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def forward_backward_propagation(self, w, b, x_train, y_train):
        m = x_train.shape[1]
        z = np.dot(w.T, x_train) + b
        y_head = self.sigmoid(z)
        y_head = np.clip(y_head, 1e-10, 1 - 1e-10)
        cost = -np.sum(y_train * np.log(y_head) + (1 - y_train) * np.log(1 - y_head)) / m
        dw = np.dot(x_train, (y_head - y_train).T) / m
        db = np.sum(y_head - y_train) / m
        gradients = {"dw": dw, "db": db}
        return cost, gradients

    def update(self, w, b, x_train, y_train):
        cost_list = []
        for i in range(self.num_iterations):
            cost, gradients = self.forward_backward_propagation(w, b, x_train, y_train)
            w = w - self.learning_rate * gradients["dw"]
            b = b - self.learning_rate * gradients["db"]
            if i % 10 == 0:
                cost_list.append(cost)
        parameters = {"w": w, "b": b}
        return parameters

    def predict(self, w, b, x):
        z = np.dot(w.T, x) + b
        y_pred = self.sigmoid(z)
        y_pred = (y_pred > 0.5).astype(int)
        return y_pred

    def fit(self, X_train, Y_train):
        dimension = X_train.shape[0]
        self.w, self.b = self.initialize_weights_and_bias(dimension)
        self.parameters = self.update(self.w, self.b, X_train, Y_train)

    def predict_class(self, X):
        y_pred = self.predict(self.parameters["w"], self.parameters["b"], X)
        return y_pred

# Define MultiNB class
class MultiNB:
    def __init__(self, alpha=1):
        self.alpha = alpha

    def _prior(self):
        P = np.zeros((self.n_classes_))
        _, self.dist = np.unique(self.y, return_counts=True)
        for i in range(self.classes_.shape[0]):
            P[i] = self.dist[i] / self.n_samples
        return P

    def fit(self, X, y):
        self.y = y
        self.n_samples, self.n_features = X.shape
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.shape[0]
        self.class_priors_ = self._prior()

        # Smoothing: Add small constant to avoid zero probabilities
        X_smoothed = X + self.alpha

        # Compute feature probabilities for each class
        self.feature_probs = np.zeros((self.n_classes_, self.n_features))
        for c in range(self.n_classes_):
            class_mask = (self.y == self.classes_[c])
            class_features = X_smoothed[class_mask]

            # Normalize probabilities
            class_totals = class_features.sum(axis=0)
            class_feature_probs = class_totals / class_totals.sum()
            self.feature_probs[c] = class_feature_probs

    def predict(self, X):
        predictions = []
        for sample in X:
            # Compute log probabilities for each class
            log_probs = np.zeros(self.n_classes_)
            for c in range(self.n_classes_):
                # Prior probability
                log_prob = np.log(self.class_priors_[c])

                # Likelihood
                for i, feature_val in enumerate(sample):
                    if feature_val > 0:
                        feature_prob = self.feature_probs[c, i]
                        log_prob += feature_val * np.log(feature_prob)

                log_probs[c] = log_prob

            # Choose class with highest log probability
            predictions.append(self.classes_[np.argmax(log_probs)])

        return np.array(predictions)

# Load models
with open('../backend/models/logistic_model.pkl', 'rb') as file:
    logistic_model = pickle.load(file)

with open('../backend/models/multi_nb_model.pkl', 'rb') as file:
    multi_nb_model = pickle.load(file)

# Load vectorizer and scaler
with open('../backend/models/vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

with open('../backend/models/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Initialize NLTK stopwords
stop_words = set(stopwords.words('english'))
port_stem = PorterStemmer()

# Preprocessing function
def preprocess_content(content):
    review = re.sub('[^a-zA-Z]', ' ', content)
    review = review.lower().split()
    review = [port_stem.stem(word) for word in review if word not in stop_words]
    return ' '.join(review)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    news = data.get('news')

    # Preprocess the input content
    processed_news = preprocess_content(news)

    # Vectorize the input
    vectorized_input = vectorizer.transform([processed_news])

    # Normalize the input
    normalized_input = scaler.transform(vectorized_input.toarray())

    # Make predictions
    logistic_result = logistic_model.predict(logistic_model.parameters['w'], logistic_model.parameters['b'], normalized_input.T)[0]  # Transpose input

    multi_nb_result = multi_nb_model.predict(normalized_input)[0]

    # Return results
    return jsonify({
        'logistic_result': 'Fake' if logistic_result == 0 else 'Real',
        'multi_nb_result': 'Fake' if multi_nb_result == 0 else 'Real'
    })


if __name__ == '__main__':
    app.run(debug=True)