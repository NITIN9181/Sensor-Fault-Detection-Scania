import joblib
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def load_model_and_preprocessor():
    model_path = 'artifacts/model.pkl'
    preprocessor_path = 'artifacts/preprocessor.pkl'

    # Load using joblib (more reliable for sklearn objects)
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    return model, preprocessor

def predict_faults(df, model, preprocessor):
    df = df.replace('na', np.nan)
    X = preprocessor.transform(df)
    preds = model.predict(X)

    # Return with row index
    return [{"row": i, "prediction": int(p)} for i, p in enumerate(preds)]



