# main.py
from src.data_loader import DataLoader


def main():
    # Initialize data loader
    data_path = 'data/raw/spam.csv'
    loader = DataLoader(data_path)

    # Load data
    df = loader.load_data()

    # Validate data
    loader.validate_data()

    # Print statistics
    loader.print_stats()

    # Show first few examples
    print("Sample messages:")
    print(df.head(10))


if __name__ == "__main__":
    main()