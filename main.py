from src.data_loader import DataLoader
from src.preprocessor import TextPreprocessor
from src.normalizer import DataNormalizer
from src.vectorizer import TextVectorizer
from src.model_trainer import ModelTrainer
from src.evaluator import ModelEvaluator  # <--- Import Evaluator
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

    # 6. Train and Evaluate
    print("\n>>> 6. Training & Visualization")
    trainer = ModelTrainer()
    evaluator = ModelEvaluator()  # Initialize Evaluator

    results = {}

    # --- Naive Bayes ---
    nb_model = trainer.train_naive_bayes(X_train_final, y_train_enc)
    acc_nb = trainer.evaluate_model('NaiveBayes', X_test_final, y_test_enc)
    results['NaiveBayes'] = acc_nb
    evaluator.plot_confusion_matrix(nb_model, X_test_final, y_test_enc, 'NaiveBayes')
    evaluator.plot_roc_curve(nb_model, X_test_final, y_test_enc, 'NaiveBayes')

    # --- Logistic Regression ---
    lr_model = trainer.train_logistic_regression(X_train_final, y_train_enc)
    acc_lr = trainer.evaluate_model('LogisticRegression', X_test_final, y_test_enc)
    results['LogisticRegression'] = acc_lr
    evaluator.plot_confusion_matrix(lr_model, X_test_final, y_test_enc, 'LogisticRegression')
    evaluator.plot_roc_curve(lr_model, X_test_final, y_test_enc, 'LogisticRegression')

    # --- Random Forest ---
    rf_model = trainer.train_random_forest(X_train_final, y_train_enc)
    acc_rf = trainer.evaluate_model('RandomForest', X_test_final, y_test_enc)
    results['RandomForest'] = acc_rf
    evaluator.plot_confusion_matrix(rf_model, X_test_final, y_test_enc, 'RandomForest')
    evaluator.plot_roc_curve(rf_model, X_test_final, y_test_enc, 'RandomForest')

    # --- Comparison ---
    evaluator.plot_model_comparison(results)

    print(f"\nTraining complete! Check the 'results/' folder for plots.")


if __name__ == "__main__":
    main()