from flask import Flask, request, jsonify, render_template
import pandas as pd
import os
import sys

from src.pipeline.predict_pipeline import PredictPipeline

app = Flask(__name__)

# Load prediction pipeline
predict_pipeline = PredictPipeline()

@app.route('/')
def home():
    return render_template('index.html')  # Optional UI page

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        df = pd.read_csv(file)

        predictions = predict_pipeline.predict(df)

        # Optional: Convert numeric predictions to labels
        label_map = {1: "Fault Detected", 0: "No Fault"}
        labels = [label_map.get(pred, str(pred)) for pred in predictions]

        return jsonify({
            "predictions": labels,
            "counts": {
                "Fault Detected": labels.count("Fault Detected"),
                "No Fault": labels.count("No Fault")
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("App running at: http://127.0.0.1:5000/")
    print("(Press CTRL+C to quit)")


    app.run(debug=True)
