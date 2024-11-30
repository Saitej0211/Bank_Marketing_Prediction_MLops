from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from google.cloud import storage
from datetime import datetime, timezone
import os
import warnings
from flask_cors import CORS
import traceback

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Define global constants
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data", "processed")
CREDENTIALS_PATH = os.path.join(PROJECT_DIR, "config", "Key.json")
BUCKET_NAME = "mlopsprojectdatabucketgrp6"
MODEL_PATH = "models/best_random_forest_model/model.pkl"

# Global variables for the model and preprocessors
model = None
preprocessors = {}

def load_preprocessing_objects(data_dir):
    """Load all preprocessing objects from the local directory."""
    preprocessors = {}
    categorical_columns = [
        "job", "marital", "education", "default", "housing",
        "loan", "contact", "month"
    ]
    try:
        for col in categorical_columns:
            with open(f"{data_dir}/{col}_label_encoder.pkl", "rb") as f:
                preprocessors[f"{col}_encoder"] = pickle.load(f)

        with open(f"{data_dir}/scaler.pkl", "rb") as f:
            preprocessors["scaler"] = pickle.load(f)

        with open(f"{data_dir}/normalizer.pkl", "rb") as f:
            preprocessors["normalizer"] = pickle.load(f)
    except Exception as e:
        print(f"Error loading preprocessing objects: {str(e)}")
        raise
    return preprocessors

def preprocess_input(input_data, preprocessors):
    """Preprocess a single row of input data."""
    try:
        df = pd.DataFrame([input_data])

        # Encode categorical columns
        for col in ["job", "marital", "education", "default", "housing", "loan", "contact", "month"]:
            df[col] = preprocessors[f"{col}_encoder"].transform([input_data[col]])

        # Normalize data
        normalized_data = preprocessors["normalizer"].transform(df)

        columns = [
            "age", "job", "marital", "education", "default", "balance",
            "housing", "loan", "contact", "day", "month", "duration",
            "campaign", "pdays", "previous"
        ]
        final_df = pd.DataFrame(normalized_data, columns=columns)
        return final_df
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        raise

def load_model_from_gcp():
    """Load the model from a GCP bucket using service account key."""
    try:
        storage_client = storage.Client.from_service_account_json(CREDENTIALS_PATH)
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(MODEL_PATH)
        model_bytes = blob.download_as_bytes()
        print("Model successfully loaded from GCP.")
        return pickle.loads(model_bytes)
    except Exception as e:
        print(f"Error loading model from GCP: {str(e)}")
        raise

@app.route("/", methods=["GET"])
def index():
    """Serve the HTML page."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Handle HTTP POST requests for predictions."""
    try:
        if request.is_json:
            input_data = request.json
        else:
            input_data = request.form.to_dict()
        
        # Convert numeric strings to actual numbers
        for key, value in input_data.items():
            if isinstance(value, str) and value.replace('.', '').isdigit():
                input_data[key] = float(value)

        # Start measuring time
        start_time = datetime.now(timezone.utc)

        # Preprocess input and make prediction
        processed_data = preprocess_input(input_data, preprocessors)
        prediction = model.predict(processed_data)

        # Measure response time
        response_time = (datetime.now(timezone.utc) - start_time).total_seconds()

        # Log metrics to BigQuery
        return jsonify({"prediction": int(prediction[0])})
    except Exception as e:
        error_message = f"An error occurred: {str(e)}\n{traceback.format_exc()}"


        return jsonify({"error": error_message}), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint to verify the service is running."""
    try:
        if model is None or not preprocessors:
            return jsonify({"status": "unhealthy", "details": "Model or preprocessors not loaded"}), 500
        return jsonify({"status": "healthy", "details": "Service is running"}), 200
    except Exception as e:
        print(f"Health check failed: {str(e)}")
        return jsonify({"status": "unhealthy", "details": str(e)}), 500

if __name__ == "__main__":
    try:
        # Lazy load preprocessors and model
        preprocessors = load_preprocessing_objects(DATA_DIR)
        model = load_model_from_gcp()
        print("Model and preprocessors loaded successfully.")
        app.run(host="0.0.0.0", port=8000, debug=True)
    except Exception as e:
        print(f"Error initializing the application: {e}")