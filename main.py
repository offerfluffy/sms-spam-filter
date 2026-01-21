from src.data_loader import DataLoader
from src.preprocessor import TextPreprocessor
from src.normalizer import DataNormalizer
from src.vectorizer import TextVectorizer
from src.model_trainer import ModelTrainer  # <--- Import this
import scipy.sparse
import pandas as pd


def main():
    # 1. Load Data
    print(">>> 1. Loading Data")
    loader = DataLoader('data/raw/spam.csv')
    df = loader.load_data()

    # 2. Preprocess Data
    print("\n>>> 2. Preprocessing")
    preprocessor = TextPreprocessor()
    df_processed = preprocessor.preprocess_dataframe(df)

    # 3. Split and Normalize Data
    print("\n>>> 3. Splitting and Normalizing")
    normalizer = DataNormalizer()
    X_train, X_test, y_train, y_test = normalizer.split_data(df_processed)
    y_train_enc, y_test_enc = normalizer.encode_labels(y_train, y_test)

    numeric_features = ['message_len', 'punct_count', 'caps_count']
    X_train_scaled, X_test_scaled = normalizer.scale_features(X_train, X_test, numeric_features)

    # 4. Vectorize Text
    print("\n>>> 4. Vectorizing Text")
    vectorizer = TextVectorizer(max_features=3000)
    tfidf_train = vectorizer.fit_transform(X_train['processed_text'])
    tfidf_test = vectorizer.transform(X_test['processed_text'])

    # 5. Combine Features
    print("\n>>> 5. Combining Features")
    X_train_final = scipy.sparse.hstack([tfidf_train, scipy.sparse.csr_matrix(X_train_scaled[numeric_features].values)])
    X_test_final = scipy.sparse.hstack([tfidf_test, scipy.sparse.csr_matrix(X_test_scaled[numeric_features].values)])

    # 6. Train and Evaluate Models
    print("\n>>> 6. Training Models")
    trainer = ModelTrainer()

    # Train Naive Bayes
    trainer.train_naive_bayes(X_train_final, y_train_enc)
    trainer.evaluate_model('NaiveBayes', X_test_final, y_test_enc)

    # Train Logistic Regression
    trainer.train_logistic_regression(X_train_final, y_train_enc)
    trainer.evaluate_model('LogisticRegression', X_test_final, y_test_enc)

    # Train Random Forest
    trainer.train_random_forest(X_train_final, y_train_enc)
    trainer.evaluate_model('RandomForest', X_test_final, y_test_enc)

    print("\nTraining complete!")


if __name__ == "__main__":
    main()