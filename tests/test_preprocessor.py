import pytest
import pandas as pd
from src.preprocessor import TextPreprocessor


class TestPreprocessor:

    @pytest.fixture
    def preprocessor(self):
        """Create a preprocessor instance for testing."""
        return TextPreprocessor()

    def test_clean_text_basic(self, preprocessor):
        """Test basic text cleaning (lowercase, punctuation)."""
        raw_text = "Hello World! This is a TEST."
        expected = "hello world this is a test"
        assert preprocessor.clean_text(raw_text) == expected

    def test_clean_text_empty(self, preprocessor):
        """Test cleaning of empty string."""
        assert preprocessor.clean_text("") == ""

    def test_clean_text_numbers(self, preprocessor):
        """Test that numbers are NOT removed (based on our logic)."""
        raw_text = "Call me at 12345!"
        expected = "call me at 12345"
        assert preprocessor.clean_text(raw_text) == expected

    def test_remove_stopwords(self, preprocessor):
        """Test removal of common stopwords."""
        # Note: 'is', 'the', 'a' are stopwords
        text = "this is the test"
        # After lemmatization/stopword removal:
        # 'this', 'is', 'the' might be removed or lemmatized.
        # Let's check a simple case we know works
        processed = preprocessor.remove_stopwords_and_lemmatize(text)
        assert "test" in processed
        assert "the" not in processed

    def test_feature_engineering(self, preprocessor):
        """Test if new feature columns are created."""
        df = pd.DataFrame({'message': ['Hello World!', 'Hi']})
        df_processed = preprocessor.add_features(df)

        assert 'message_len' in df_processed.columns
        assert 'punct_count' in df_processed.columns
        assert 'caps_count' in df_processed.columns

        # Check specific values for "Hello World!"
        # Length: 12, Punctuation: 1 (!), Caps: 2 (H, W)
        row0 = df_processed.iloc[0]
        assert row0['message_len'] == 12
        assert row0['punct_count'] == 1
        assert row0['caps_count'] == 2