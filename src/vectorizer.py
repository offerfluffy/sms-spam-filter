import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse


class TextVectorizer:
    """Class for vectorizing text data using TF-IDF."""

    # Update __init__ to accept min_df and max_df
    def __init__(self, max_features: int = 3000, min_df=5, max_df=0.7):
        """
        Initialize vectorizer.

        Args:
            max_features: Maximum number of words to keep
            min_df: Minimum document frequency (int for count, float for ratio)
            max_df: Maximum document frequency (float for ratio)
        """
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,  # <--- Use the variable
            max_df=max_df,  # <--- Use the variable
            ngram_range=(1, 2)
        )

    def fit_transform(self, X_train_text: pd.Series) -> scipy.sparse.csr_matrix:
        """Fit vectorizer on training text and transform it."""
        return self.tfidf.fit_transform(X_train_text)

    def transform(self, X_test_text: pd.Series) -> scipy.sparse.csr_matrix:
        """Transform test text using fitted vectorizer."""
        return self.tfidf.transform(X_test_text)