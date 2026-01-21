# main.py
from src.data_loader import DataLoader
from src.preprocessor import TextPreprocessor


def main():
    # 1. Load Data
    loader = DataLoader('data/raw/spam.csv')
    df = loader.load_data()
    loader.validate_data()

    # 2. Preprocess Data
    preprocessor = TextPreprocessor()

    print("\nProcessing data...")
    df_processed = preprocessor.preprocess_dataframe(df)

    # Show results
    print("\nOriginal vs Processed:")
    print(df_processed[['message', 'processed_text']].head())

    print("\nExtracted Features:")
    print(df_processed[['message_len', 'punct_count', 'caps_count']].head())

    # Save processed data (Checkpoint)
    df_processed.to_csv('data/processed/spam_processed.csv', index=False)
    print("\nProcessed data saved to data/processed/spam_processed.csv")


if __name__ == "__main__":
    main()