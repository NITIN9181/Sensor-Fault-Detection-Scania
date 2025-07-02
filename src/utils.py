import os
import sys
import pickle
from src.exception import SensorException

def save_object(file_path: str, obj) -> None:
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise SensorException(e, sys)


def load_object(file_path: str):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise SensorException(e, sys)
