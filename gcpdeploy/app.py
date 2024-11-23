from flask import Flask, request, jsonify
import pickle
import pandas as pd
from google.cloud import storage
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Initialize Flask app
app = Flask(__name__)

# Set paths for data and Google Cloud credentials
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data", "processed")
KEY_PATH = os.path.join(PROJECT_DIR, "config", "key.json")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = KEY_PATH

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
        with open(f"{data_dir}\{col}_label_encoder.pkl", "rb") as f:
            preprocessors[f"{col}_encoder"] = pickle.load(f)

    # Load scaler and normalizer
    with open(f"{data_dir}\scaler.pkl", "rb") as f:
        preprocessors["scaler"] = pickle.load(f)

    with open(f"{data_dir}\\normalizer.pkl", "rb") as f:
        preprocessors["normalizer"] = pickle.load(f)

    return preprocessors

def preprocess_input(input_data, preprocessors):
    """Preprocess a single row of input data"""
    # Create DataFrame with single row
    df = pd.DataFrame([input_data])

    # Process categorical columns one by one
    for col in ["job", "marital", "education", "default", "housing", "loan", "contact", "month"]:
        df[col] = preprocessors[f"{col}_encoder"].transform([input_data[col]])

    # Normalize the entire dataset
    normalized_data = preprocessors["normalizer"].transform(df)

    # Convert to DataFrame with correct column order
    columns = [
        "age", "job", "marital", "education", "default", "balance",
        "housing", "loan", "contact", "day", "month", "duration",
        "campaign", "pdays", "previous"
    ]
    final_df = pd.DataFrame(normalized_data, columns=columns)
    return final_df

def load_model_from_gcp():
    """Load model from GCP bucket"""
    storage_client = storage.Client()
    bucket = storage_client.bucket("mlopsprojectdatabucketgrp6")
    blob = bucket.blob("models/best_random_forest_model/model.pkl")
    model_bytes = blob.download_as_bytes()
    return pickle.loads(model_bytes)

@app.route("/predict", methods=["POST"])
def predict():
    """Handle HTTP POST requests for predictions"""
    try:
        # Get input data from POST request
        input_data = request.json

        # Preprocess input data
        processed_data = preprocess_input(input_data, preprocessors)

        # Make prediction
        prediction = model.predict(processed_data)

        # Return prediction as JSON
        return jsonify({"prediction": int(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint to verify the service is running."""
    try:
        # Perform a basic health check (e.g., check if model and preprocessors are loaded)
        if model is None or not preprocessors:
            return jsonify({"status": "unhealthy", "details": "Model or preprocessors not loaded"}), 500

        # Optional: Add more health checks if needed
        return jsonify({"status": "healthy", "details": "Service is running"}), 200
    except Exception as e:
        return jsonify({"status": "unhealthy", "details": str(e)}), 500

# Load preprocessing objects and model at startup
if __name__ == "__main__":
    try:
        # Load preprocessing objects
        preprocessors = load_preprocessing_objects(DATA_DIR)

        # Load model from GCP
        model = load_model_from_gcp()

        # Run Flask app
        app.run(host="0.0.0.0", port=8000, debug=True)
    except Exception as e:
        print(f"Error initializing the application: {e}")
