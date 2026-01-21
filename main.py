from src.data_loader import DataLoader
from src.preprocessor import TextPreprocessor
from src.normalizer import DataNormalizer
from src.vectorizer import TextVectorizer
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

    # Split into Train/Test
    X_train, X_test, y_train, y_test = normalizer.split_data(df_processed)

    # Encode Labels (ham=0, spam=1)
    y_train_enc, y_test_enc = normalizer.encode_labels(y_train, y_test)
    print(f"Train set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")

    # Scale numerical features
    numeric_features = ['message_len', 'punct_count', 'caps_count']
    X_train, X_test = normalizer.scale_features(X_train, X_test, numeric_features)

    # 4. Vectorize Text
    print("\n>>> 4. Vectorizing Text")
    vectorizer = TextVectorizer(max_features=3000)

    # Create TF-IDF matrices
    tfidf_train = vectorizer.fit_transform(X_train['processed_text'])
    tfidf_test = vectorizer.transform(X_test['processed_text'])

    print(f"TF-IDF Matrix shape: {tfidf_train.shape}")

    # 5. Combine Features (Numerical + TF-IDF)
    # We combine the sparse TF-IDF matrix with our scaled numerical features
    from scipy.sparse import hstack

    X_train_final = hstack([tfidf_train, scipy.sparse.csr_matrix(X_train[numeric_features].values)])
    X_test_final = hstack([tfidf_test, scipy.sparse.csr_matrix(X_test[numeric_features].values)])

    print(f"Final Training Data Shape: {X_train_final.shape}")
    print("\nData preparation complete! Ready for training.")


if __name__ == "__main__":
    main()