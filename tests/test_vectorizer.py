import pytest
import pandas as pd
import scipy.sparse
from src.vectorizer import TextVectorizer


class TestVectorizer:

    @pytest.fixture
    def vectorizer(self):
        # Override default min_df=5 with min_df=1 for small test data
        return TextVectorizer(max_features=10, min_df=1, max_df=1.0)

    def test_vectorizer_output_shape(self, vectorizer):
        """Test if vectorizer returns correct matrix shape."""
        # Create dummy data
        corpus = pd.Series([
            "spam text message",
            "ham normal message",
            "spam offer free"
        ])

        # Fit transform
        matrix = vectorizer.fit_transform(corpus)

        # Check it is a sparse matrix
        assert scipy.sparse.issparse(matrix)

        # Check dimensions: 3 documents, at most 10 features
        assert matrix.shape[0] == 3
        assert matrix.shape[1] <= 10

    def test_vectorizer_transform(self, vectorizer):
        """Test if transform works on new data."""
        train_data = pd.Series(["apple banana", "orange fruit"])
        test_data = pd.Series(["apple fruit"])

        vectorizer.fit_transform(train_data)
        result = vectorizer.transform(test_data)

        assert scipy.sparse.issparse(result)
        assert result.shape[0] == 1