import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple


class DataNormalizer:
    """Class for normalizing features and encoding labels."""

    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()

    def split_data(self, df: pd.DataFrame, target_col: str = 'label',
                   test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and test sets.

        Args:
            df: processed DataFrame
            target_col: name of the target column
            test_size: proportion of dataset to include in the test split

        Returns:
            X_train, X_test, y_train, y_test
        """
        X = df.drop(columns=[target_col])
        y = df[target_col]

        return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    def encode_labels(self, y_train: pd.Series, y_test: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Encode target labels (e.g., 'spam' -> 1, 'ham' -> 0).
        """
        # Fit on training data only to prevent data leakage
        y_train_enc = self.label_encoder.fit_transform(y_train)
        y_test_enc = self.label_encoder.transform(y_test)

        return y_train_enc, y_test_enc

    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                       numeric_cols: list) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Scale numerical features using StandardScaler.
        """
        # Create copies to avoid SettingWithCopyWarning
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()

        # Fit scaler on training data
        self.scaler.fit(X_train[numeric_cols])

        # Transform both train and test
        X_train_scaled[numeric_cols] = self.scaler.transform(X_train[numeric_cols])
        X_test_scaled[numeric_cols] = self.scaler.transform(X_test[numeric_cols])

        return X_train_scaled, X_test_scaled