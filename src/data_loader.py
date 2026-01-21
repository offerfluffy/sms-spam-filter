import pandas as pd
import os
from typing import Tuple


class DataLoader:
    """Class for loading and validating SMS spam dataset."""
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.data = None

    def load_data(self) -> pd.DataFrame:
        """
        Load data from CSV file.

        Returns:
            DataFrame with loaded data
        """
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"File not found: {self.filepath}")

        # The SMS Spam Collection dataset has specific encoding
        self.data = pd.read_csv(self.filepath, encoding='latin-1')

        # The dataset has extra columns, keep only first two
        self.data = self.data.iloc[:, :2]
        self.data.columns = ['label', 'message']

        print(f"Data loaded successfully: {len(self.data)} rows")
        return self.data

    def validate_data(self) -> bool:
        """
        Validate the loaded data.

        Returns:
            True if data is valid, raises exception otherwise
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        # Check for required columns
        required_columns = ['label', 'message']
        if not all(col in self.data.columns for col in required_columns):
            raise ValueError(f"Missing required columns: {required_columns}")

        # Check for missing values
        if self.data.isnull().any().any():
            print("Warning: Dataset contains missing values")
            print(self.data.isnull().sum())

        # Check label values
        unique_labels = self.data['label'].unique()
        print(f"Unique labels: {unique_labels}")

        if not all(label in ['ham', 'spam'] for label in unique_labels):
            raise ValueError("Labels must be 'ham' or 'spam'")

        print("Data validation passed ✓")
        return True

    def get_basic_stats(self) -> dict:
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        stats = {
            'total_messages': len(self.data),
            'spam_count': len(self.data[self.data['label'] == 'spam']),
            'ham_count': len(self.data[self.data['label'] == 'ham']),
            'spam_percentage': (len(self.data[self.data['label'] == 'spam']) / len(self.data)) * 100,
            'avg_message_length': self.data['message'].str.len().mean(),
            'max_message_length': self.data['message'].str.len().max(),
            'min_message_length': self.data['message'].str.len().min()
        }

        return stats

    def print_stats(self):
        stats = self.get_basic_stats()
        print("\n" + "=" * 50)
        print("DATASET STATISTICS")
        print("=" * 50)
        print(f"Total messages: {stats['total_messages']}")
        print(f"Spam messages: {stats['spam_count']} ({stats['spam_percentage']:.2f}%)")
        print(f"Ham messages: {stats['ham_count']} ({100 - stats['spam_percentage']:.2f}%)")
        print(f"\nMessage length:")
        print(f"  Average: {stats['avg_message_length']:.2f} characters")
        print(f"  Min: {stats['min_message_length']} characters")
        print(f"  Max: {stats['max_message_length']} characters")
        print("=" * 50 + "\n")