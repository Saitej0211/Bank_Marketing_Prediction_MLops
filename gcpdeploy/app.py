from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from google.cloud import storage
import os
import warnings
import traceback
from flask_cors import CORS

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Set paths for data
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data", "processed")

# Global variables for model and preprocessors
model = None
preprocessors = {}

def load_preprocessing_objects(data_dir):
    """Load all preprocessing objects from local directory"""
    preprocessors = {}
    categorical_columns = [
        "job", "marital", "education", "default", "housing",
        "loan", "contact", "month"
    ]
    for col in categorical_columns:
        with open(f"{data_dir}/{col}_label_encoder.pkl", "rb") as f:
            preprocessors[f"{col}_encoder"] = pickle.load(f)

    with open(f"{data_dir}/scaler.pkl", "rb") as f:
        preprocessors["scaler"] = pickle.load(f)

    with open(f"{data_dir}/normalizer.pkl", "rb") as f:
        preprocessors["normalizer"] = pickle.load(f)

    return preprocessors

def preprocess_input(input_data, preprocessors):
    """Preprocess a single row of input data"""
    df = pd.DataFrame([input_data])

    for col in ["job", "marital", "education", "default", "housing", "loan", "contact", "month"]:
        df[col] = preprocessors[f"{col}_encoder"].transform([input_data[col]])

    normalized_data = preprocessors["normalizer"].transform(df)

    columns = [
        "age", "job", "marital", "education", "default", "balance",
        "housing", "loan", "contact", "day", "month", "duration",
        "campaign", "pdays", "previous"
    ]
    final_df = pd.DataFrame(normalized_data, columns=columns)
    return final_df

def load_model_from_gcp():
    """Load model from GCP bucket using Application Default Credentials"""
    storage_client = storage.Client()
    bucket = storage_client.bucket("mlopsprojectdatabucketgrp6")
    blob = bucket.blob("models/best_random_forest_model/model.pkl")
    model_bytes = blob.download_as_bytes()
    return pickle.loads(model_bytes)

@app.route("/", methods=["GET"])
def index():
    """Serve the HTML page."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = request.json
        processed_data = preprocess_input(input_data, preprocessors)
        prediction = model.predict(processed_data)
        return jsonify({"prediction": int(prediction[0])})
    except Exception as e:
        error_message = f"An error occurred: {str(e)}\n{traceback.format_exc()}"
        print(error_message)  # Log the error on the server side
        return jsonify({"error": error_message}), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint to verify the service is running."""
    try:
        if model is None or not preprocessors:
            return jsonify({"status": "unhealthy", "details": "Model or preprocessors not loaded"}), 500
        return jsonify({"status": "healthy", "details": "Service is running"}), 200
    except Exception as e:
        return jsonify({"status": "unhealthy", "details": str(e)}), 500

if __name__ == "__main__":
    try:
        preprocessors = load_preprocessing_objects(DATA_DIR)
        model = load_model_from_gcp()
        app.run(host="0.0.0.0", port=8000, debug=True)
    except Exception as e:
        print(f"Error initializing the application: {e}")