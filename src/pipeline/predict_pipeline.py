import numpy as np
import pandas as pd
import joblib
import os
import sys

from src.exception import SensorException
from src.logger import logging

class PredictPipeline:
    def __init__(self):
        try:
            # Define model and preprocessor paths
            self.model_path = os.path.join("artifacts", "model.pkl")
            self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            logging.info("Loading model and preprocessor for prediction...")
            self.model = joblib.load(self.model_path)
            self.preprocessor = joblib.load(self.preprocessor_path)

        except Exception as e:
            raise SensorException(e, sys)

    def predict(self, input_df: pd.DataFrame):
        try:
            logging.info("Transforming input data using preprocessor...")
            transformed_data = self.preprocessor.transform(input_df)

            logging.info("Generating predictions using trained model...")
            predictions = self.model.predict(transformed_data)

            return predictions

        except Exception as e:
            raise SensorException(e, sys)
