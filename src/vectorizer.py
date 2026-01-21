import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse


class TextVectorizer:
    """Class for vectorizing text data using TF-IDF."""

    def __init__(self, max_features: int = 3000):
        """
        Initialize vectorizer.

        Args:
            max_features: Maximum number of words to keep (vocabulary size)
        """
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            min_df=5,  # Ignore words that appear in fewer than 5 docs
            max_df=0.7,  # Ignore words that appear in > 70% of docs
            ngram_range=(1, 2)  # Use single words and pairs of words (bi-grams)
        )

    def fit_transform(self, X_train_text: pd.Series) -> scipy.sparse.csr_matrix:
        """Fit vectorizer on training text and transform it."""
        return self.tfidf.fit_transform(X_train_text)

    def transform(self, X_test_text: pd.Series) -> scipy.sparse.csr_matrix:
        """Transform test text using fitted vectorizer."""
        return self.tfidf.transform(X_test_text)