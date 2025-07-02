import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
import joblib

from src.exception import SensorException
from src.logger import logger

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()
        
    def get_data_transformer_object(self):
        """
        Returns a ColumnTransformer object with scaling and imputation.
        """
        try:
            # Identify numerical columns (all except 'class')
            numerical_columns = [
                col for col in pd.read_csv(os.path.join("artifacts", "train.csv")).columns
                if col != "class"
            ]

            # Pipeline for numeric features
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler())
            ])

            # Apply only to numeric columns
            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns)
            ])

            return preprocessor

        except Exception as e:
            raise SensorException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            logger.info("Starting data transformation.")

            # Load train and test data
            train_df = pd.read_csv(train_path).replace('na', np.nan)
            test_df = pd.read_csv(test_path).replace('na', np.nan)

            logger.info("Train and test data loaded.")

            # Separate input features and target
            X_train = train_df.drop("class", axis=1)
            y_train = train_df["class"].replace({"pos": 1, "neg": 0}).astype(int)

            X_test = test_df.drop("class", axis=1)
            y_test = test_df["class"].replace({"pos": 1, "neg": 0}).astype(int)

            # Get the preprocessing pipeline
            preprocessor = self.get_data_transformer_object()

            # Fit and transform on train, transform only on test
            X_train_scaled = preprocessor.fit_transform(X_train)
            X_test_scaled = preprocessor.transform(X_test)

            # Save the preprocessor object
            joblib.dump(preprocessor, self.transformation_config.preprocessor_obj_file_path)
            logger.info(f"Preprocessor object saved at {self.transformation_config.preprocessor_obj_file_path}")

            logger.info("Data transformation completed.")
            return (
                X_train_scaled, y_train.values,
                X_test_scaled, y_test.values,
                self.transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise SensorException(e, sys)
