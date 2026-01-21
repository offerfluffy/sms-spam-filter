import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


class ModelTrainer:
    """Class for training and evaluating machine learning models."""

    def __init__(self):
        """Initialize dictionary to store trained models."""
        self.models = {}

    def train_naive_bayes(self, X_train, y_train):
        """Train Multinomial Naive Bayes classifier."""
        print("Training Naive Bayes...")
        model = MultinomialNB()
        model.fit(X_train, y_train)
        self.models['NaiveBayes'] = model
        return model

    def train_logistic_regression(self, X_train, y_train):
        """Train Logistic Regression classifier."""
        print("Training Logistic Regression...")
        # max_iter=1000 ensures the model has enough time to find the best solution
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        self.models['LogisticRegression'] = model
        return model

    def train_random_forest(self, X_train, y_train):
        """Train Random Forest classifier."""
        print("Training Random Forest...")
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        self.models['RandomForest'] = model
        return model

    def evaluate_model(self, model_name, X_test, y_test):
        """
        Evaluate a specific trained model.

        Args:
            model_name: Key name of the model in self.models
            X_test: Test features
            y_test: True test labels
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Train it first.")

        model = self.models[model_name]
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n--- Results for {model_name} ---")
        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        return accuracy