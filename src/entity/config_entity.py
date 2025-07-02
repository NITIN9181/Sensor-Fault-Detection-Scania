from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    raw_data_path: str
    train_data_path: str
    test_data_path: str

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str
