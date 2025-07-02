from src.components.data_ingestion import DataIngestion

if __name__ == "__main__":
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()
    print(f"Train file: {train_path}")
    print(f"Test file: {test_path}")