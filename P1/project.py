import os
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load datasets
dataset_dir = './'
for filename in os.listdir(dataset_dir):
    if filename.endswith('.csv'):
        var_name = os.path.splitext(filename)[0]
        globals()[var_name] = pd.read_csv(os.path.join(dataset_dir, filename))
        print(f"Loaded file: {filename} into variable: {var_name}")

# Reduce dataset to half. I did this because with whole dataset, the program stopped due to insufficient memory
news_data = train.sample(frac=0.5, random_state=42)  # Randomly select half of the data
print(f"Dataset reduced to half. New shape: {news_data.shape}")

# Preprocessing Data
news_data = news_data.fillna('')  # Fill missing values with empty strings
news_data['content'] = news_data['author'] + ' ' + news_data['title']  # Combine author and title

# Stemming Function
port_stem = PorterStemmer()

def stemming(content):
    review = re.sub('[^a-zA-Z]', ' ', content)  # Remove non-alphabetic characters
    review = review.lower().split()  # Convert to lowercase and split into words
    review = [port_stem.stem(word) for word in review if word not in stop_words]
    return ' '.join(review)

news_data['content'] = news_data['content'].apply(stemming)  # Apply stemming
X = news_data['content'].values  # Features
Y = news_data['label'].values  # Labels

# TF-IDF 
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X).toarray()

# Normalize 
scaler = StandardScaler()
X = scaler.fit_transform(X)


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

# Reshape Y_train and Y_test to be column vectors
Y_train = Y_train.reshape(-1, 1)  
Y_test = Y_test.reshape(-1, 1)    


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
                print(f"Cost after iteration {i}: {cost}")
        plt.plot(range(0, self.num_iterations, 10), cost_list)
        plt.xlabel("Number of Iterations")
        plt.ylabel("Cost")
        plt.title("Cost Reduction Over Iterations")
        plt.show()
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


logistic_model = LogisticRegressionManual(learning_rate=0.01, num_iterations=1000)
logistic_model.fit(X_train.T, Y_train.T) 

# Predictions and Accuracy
logistic_train_pred = logistic_model.predict_class(X_train.T)  # Transpose X_train
logistic_test_pred = logistic_model.predict_class(X_test.T)  # Transpose X_test
logistic_train_acc = 100 - np.mean(np.abs(logistic_train_pred - Y_train.T)) * 100
logistic_test_acc = 100 - np.mean(np.abs(logistic_test_pred - Y_test.T)) * 100
print(f"Logistic Regression - Training Accuracy: {logistic_train_acc}%")
print(f"Logistic Regression - Test Accuracy: {logistic_test_acc}%")


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


multi_nb_model = MultiNB(alpha=1)
multi_nb_model.fit(X_train, Y_train.ravel())  # Flatten Y_train


multi_nb_train_pred = multi_nb_model.predict(X_train)
multi_nb_test_pred = multi_nb_model.predict(X_test)
multi_nb_train_acc = 100 - np.mean(np.abs(multi_nb_train_pred - Y_train.ravel())) * 100
multi_nb_test_acc = 100 - np.mean(np.abs(multi_nb_test_pred - Y_test.ravel())) * 100
print(f"Multinomial Naive Bayes - Training Accuracy: {multi_nb_train_acc}%")
print(f"Multinomial Naive Bayes - Test Accuracy: {multi_nb_test_acc}%")


#plot confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

# Plot confusion matrix for Logistic Regression
plot_confusion_matrix(Y_test.ravel(), logistic_test_pred.ravel(), "Logistic Regression - Confusion Matrix")

# Plot confusion matrix for Multinomial Naive Bayes
plot_confusion_matrix(Y_test.ravel(), multi_nb_test_pred, "Multinomial Naive Bayes - Confusion Matrix")


# Save models and vectorizer to disk
os.makedirs('backend/models', exist_ok=True)

# Save Logistic Regression model
with open('backend/models/logistic_model.pkl', 'wb') as f:
    pickle.dump(logistic_model, f)

# Save Multinomial Naive Bayes model
with open('backend/models/multi_nb_model.pkl', 'wb') as f:
    pickle.dump(multi_nb_model, f)

# Save TF-IDF Vectorizer
with open('backend/models/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# Save StandardScaler
with open('backend/models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Models and vectorizer saved successfully.")
