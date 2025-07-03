# pipeline/training_pipeline.py

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging
from src.exception import SensorException

import numpy as np

def run_training_pipeline():
    try:
        logging.info("Starting the data ingestion process...")
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()

        logging.info("Starting the data transformation process...")
        transformation = DataTransformation()
        X_train, y_train, X_test, y_test, preprocessor_path = transformation.initiate_data_transformation(train_path, test_path)

        # Combine features and labels into a single array (last column is label)
        train_array = np.c_[X_train, y_train]
        test_array = np.c_[X_test, y_test]

        logging.info("Starting the model training process...")
        trainer = ModelTrainer()
        model_score = trainer.initiate_model_trainer(train_array, test_array)

        logging.info(f"Model training complete. Accuracy: {model_score}")
        print(f"Model training complete. Accuracy: {model_score}")
        return model_score

    except Exception as e:
        logging.error("Pipeline failed due to: %s", e)
        print(f"Pipeline failed due to: {e}")
        raise e
