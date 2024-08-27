from flask import Flask, request, render_template, jsonify
import pandas as pd
import os
import sys
import joblib

from code.feature_engineering import extract_features
from code.scale_features import scale_features_from_stored

app = Flask(__name__)

# Dictionary to store loaded models
loaded_models = {}

def load_model(model_name):
    model_filename = f'model/{model_name}.pkl'
    model_path = os.path.join(model_filename)
    if model_name not in loaded_models:
        if os.path.exists(model_path):
            loaded_models[model_name] = joblib.load(model_path)
        else:
            raise FileNotFoundError(f"Model file {model_filename} not found.")
    return loaded_models[model_name]

@app.route('/', methods=['GET', 'POST'])
def index():
    # List all .pkl files in the model directory
    model_files = [f[:-4] for f in os.listdir(os.path.join('model')) if f.endswith('.pkl')]
    prediction = None
    if request.method == 'POST':
        word = request.form['word']
        model_name = request.form['model']
        prediction = predict_password_likelihood(word, model_name)
    return render_template('index.html', prediction=prediction, models=model_files)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    model_name = data.get('model')
    words = data.get('words', [])
    predictions = []

    for word in words:
        score = predict_password_likelihood(word, model_name)
        predictions.append({'word': word, 'score': score})

    return jsonify({'predictions': predictions})

def predict_password_likelihood(word, model_name):
    # Load the selected model
    model = load_model(model_name)
    
    # Extract features from the word
    features = extract_features(word)
    
    # Convert the dictionary of features to a pandas DataFrame
    features_df = pd.DataFrame([features])
    
    # Scale the extracted features
    scaled_features_df = scale_features_from_stored(features_df)
    
    # Execute the model to get the prediction
    prediction_proba = model.predict_proba(scaled_features_df)[0, 1]
    
    return float(prediction_proba)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)