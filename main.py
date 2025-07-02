from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

if __name__ == "__main__":
    # Ingestion
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    # Transformation
    transformation = DataTransformation()
    (
        X_train, y_train,
        X_test, y_test,
        preprocessor_path
    ) = transformation.initiate_data_transformation(train_path, test_path)

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"Preprocessor saved at: {preprocessor_path}")
