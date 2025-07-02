from flask import Flask, request, jsonify, render_template
import pandas as pd
from flask_app.utils import load_model_and_preprocessor, predict_faults

app = Flask(__name__)

# Load model and preprocessor once when the app starts
model, preprocessor = load_model_and_preprocessor()

@app.route('/')
def home():
    return render_template('index.html')  # optional HTML UI

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        data = pd.read_csv(file)

        predictions = predict_faults(data, model, preprocessor)
        return jsonify({"predictions": predictions})

    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
