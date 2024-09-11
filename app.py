from flask import Flask, request, render_template, jsonify
import pandas as pd
import os
import sys
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, TFBertForSequenceClassification
from app_code.feature_engineering import extract_features
from app_code.scale_features import scale_features_from_stored

app = Flask(__name__)

# Dictionary to store loaded models
loaded_models = {}

def load_model(model_name):
    
    model_path = os.path.join("model", model_name)
    
    if 'bert.config' in model_name.lower():
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Config file {model_path} not found.")
        
        with open(model_path) as f:
            path = f.read().strip()
            path = os.path.join("model", path)
            
        print(f"Loading model from path: {path}")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model path {path} specified in config file not found.")
        
        model = TFBertForSequenceClassification.from_pretrained(path)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        loaded_models[model_name] = {
            'model': model,
            'tokenizer': tokenizer
        }
        return loaded_models[model_name]
        
    model_path = os.path.join("model/",model_name)
    # model_path = os.path.join(model_filename)
    # keras_model_path = f'model/{model_name}.keras'
    
    if model_name not in loaded_models:
        if os.path.exists(model_path) and model_path.endswith('.pkl'):
            loaded_models[model_name] = joblib.load(model_path)
        
        elif os.path.exists(model_path) and model_path.endswith('.h5'):
            loaded_models[model_name] = {
                'model': TFBertForSequenceClassification.from_pretrained(model_path),
                'tokenizer': BertTokenizer.from_pretrained('bert-base-uncased')
            }
            
        elif os.path.exists(model_path) and model_path.endswith('.keras'):
            loaded_models[model_name] = tf.keras.models.load_model(model_path)
        else:
            raise FileNotFoundError(f"Model file {model_path} not found.")
    return loaded_models[model_name]

def load_model_and_predict(words, model):
    # Initialize the tokenizer
    tokenizer = Tokenizer(char_level=True, num_words=50000)
    
    # Fit the tokenizer on the input words (for character-level tokenization)
    tokenizer.fit_on_texts(words)
    
    # Convert words to sequences
    sequences = tokenizer.texts_to_sequences(words)
    
    # Pad the sequences to ensure uniform length
    padded_sequences = pad_sequences(sequences, maxlen=64, padding='post')

    # Predict the scores for each word
    predictions = model.predict(padded_sequences).flatten()

    return predictions

def load_model_and_predict_with_features(words, model):
    # Initialize the tokenizer
    tokenizer = Tokenizer(char_level=True, num_words=50000)
    
    # Fit the tokenizer on the input words (for character-level tokenization)
    tokenizer.fit_on_texts(words)
    
    # Convert words to sequences
    sequences = tokenizer.texts_to_sequences(words)

    # Get the expected input shape of the model
    expected_input_shape = model.input_shape[0][1] if isinstance(model.input_shape, list) else model.input_shape[1]
    
    # Pad the sequences to ensure uniform length
    # padded_sequences = pad_sequences(sequences, maxlen=64, padding='post')
    padded_sequences = pad_sequences(sequences, maxlen=expected_input_shape, padding='post')
    
    predictions = []
    for word in words:
        # Extract features from the word
        features = extract_features(word)
        
        # Convert the dictionary of features to a pandas DataFrame
        features_df = pd.DataFrame([features])
        
        # Scale the extracted features
        scaled_features_df = scale_features_from_stored(features_df)
        
        # Predict the scores for each word
        prediction = model.predict([padded_sequences, scaled_features_df]).flatten()
        predictions.append(prediction[0])
    
    return predictions



@app.route('/', methods=['GET', 'POST'])
def index():
    # List all .pkl and .keras files in the model directory
    model_files = sorted([f for f in os.listdir(os.path.join('model')) if f.endswith('.pkl') or f.endswith('.keras') or f.endswith('bert.config')])
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
    model_data = load_model(model_name)
    
    if isinstance(model_data, dict) and 'model' in model_data and 'tokenizer' in model_data:
        model = model_data['model']
        tokenizer = model_data['tokenizer']
    else:
        model = model_data
    
    if isinstance(model, tf.keras.Model):
        # If the model is a Keras model, use the Keras prediction function
        if 'features' in model_name.lower():
            predictions = load_model_and_predict_with_features([word], model)
        else:
            predictions = load_model_and_predict([word], model)
        
        return float(predictions[0])
    elif 'bert' in model_name.lower():
        # BERT model prediction
        encoding = tokenizer([word], truncation=True, padding='max_length', max_length=64, return_tensors='tf')
        logits = model(encoding)['logits']
        prediction = tf.nn.sigmoid(logits).numpy().flatten()
        probability = prediction[0]
        
        return float(probability)
    else:
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