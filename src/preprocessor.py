import pandas as pd
import re
import string
import nltk
import ssl  # <--- Add this import
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- ADD THIS BLOCK TO FIX SSL ERROR ON MAC ---
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


# ----------------------------------------------

class TextPreprocessor:
    """Class for cleaning and preprocessing text data."""

    def __init__(self):
        """Initialize preprocessor and download necessary NLTK data."""
        # Download NLTK resources if not present
        try:
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            print("Downloading NLTK resources...")
            nltk.download('stopwords')
            nltk.download('wordnet')
            nltk.download('omw-1.4')

        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text: str) -> str:
        # ... (rest of your code remains exactly the same)
        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Remove extra whitespace
        text = text.strip()

        return text

    def remove_stopwords_and_lemmatize(self, text: str) -> str:
        # ... (rest of your code remains exactly the same)
        tokens = text.split()

        processed_tokens = [
            self.lemmatizer.lemmatize(word)
            for word in tokens
            if word not in self.stop_words
        ]

        return ' '.join(processed_tokens)

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # ... (rest of your code remains exactly the same)
        df_stats = df.copy()

        df_stats['message_len'] = df_stats['message'].apply(len)

        df_stats['punct_count'] = df_stats['message'].apply(
            lambda x: len([c for c in x if c in string.punctuation])
        )

        df_stats['caps_count'] = df_stats['message'].apply(
            lambda x: len([c for c in x if c.isupper()])
        )

        return df_stats

    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str = 'message') -> pd.DataFrame:
        # ... (rest of your code remains exactly the same)
        print("Starting preprocessing...")
        df_processed = df.copy()

        # 1. Feature Engineering
        df_processed = self.add_features(df_processed)

        # 2. Basic Cleaning
        df_processed['processed_text'] = df_processed[text_column].apply(self.clean_text)

        # 3. Stopwords & Lemmatization
        df_processed['processed_text'] = df_processed['processed_text'].apply(self.remove_stopwords_and_lemmatize)

        print("Preprocessing complete.")
        return df_processed