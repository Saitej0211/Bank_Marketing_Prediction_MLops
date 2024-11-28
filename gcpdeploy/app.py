from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from google.cloud import storage, bigquery
import os
import warnings
import traceback
import logging
from flask_cors import CORS
from datetime import datetime, timezone

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Set paths for data
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data", "processed")

# Global variables for model and preprocessors
model = None
preprocessors = {}

# BigQuery table ID for logging
BIGQUERY_TABLE_ID = "dvc-lab-439300.model_metrics_dataset.metrics_log"  # Replace with your project info

def log_to_bigquery(endpoint, input_data, prediction, response_time, status):
    """Log metrics to BigQuery"""
    client = bigquery.Client()
    rows_to_insert = [
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "endpoint": endpoint,
            "input_data": str(input_data),
            "prediction": str(prediction),
            "response_time": response_time,
            "status": status,
        }
    ]
    errors = client.insert_rows_json(BIGQUERY_TABLE_ID, rows_to_insert)
    if errors:
        logger.error(f"Failed to log to BigQuery: {errors}")
    else:
        logger.info(f"Logged metrics to BigQuery: {rows_to_insert}")

def load_preprocessing_objects(data_dir):
    """Load all preprocessing objects from local directory"""
    preprocessors = {}
    categorical_columns = [
        "job", "marital", "education", "default", "housing",
        "loan", "contact", "month"
    ]
    for col in categorical_columns:
        try:
            with open(f"{data_dir}/{col}_label_encoder.pkl", "rb") as f:
                preprocessors[f"{col}_encoder"] = pickle.load(f)
            logger.info(f"Loaded encoder for column: {col}")
        except Exception as e:
            logger.error(f"Error loading encoder for {col}: {str(e)}")
            raise

    try:
        with open(f"{data_dir}/scaler.pkl", "rb") as f:
            preprocessors["scaler"] = pickle.load(f)
        logger.info("Loaded scaler")
    except Exception as e:
        logger.error(f"Error loading scaler: {str(e)}")
        raise

    try:
        with open(f"{data_dir}/normalizer.pkl", "rb") as f:
            preprocessors["normalizer"] = pickle.load(f)
        logger.info("Loaded normalizer")
    except Exception as e:
        logger.error(f"Error loading normalizer: {str(e)}")
        raise

    return preprocessors

def preprocess_input(input_data, preprocessors):
    """Preprocess a single row of input data"""
    df = pd.DataFrame([input_data])

    # Apply preprocessing steps (encoding categorical features)
    for col in ["job", "marital", "education", "default", "housing", "loan", "contact", "month"]:
        df[col] = preprocessors[f"{col}_encoder"].transform([input_data[col]])

    # Normalize the data
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
    bucket = storage_client.bucket("mlopsprojectdatabucketgrp6")  # Update with your bucket name
    blob = bucket.blob("models/best_random_forest_model/model.pkl")  # Update with the model path
    model_bytes = blob.download_as_bytes()
    return pickle.loads(model_bytes)

@app.route("/", methods=["GET"])
def index():
    """Serve the HTML page."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Handle HTTP POST requests for predictions"""
    try:
        input_data = request.json
        logger.info(f"Received input data: {input_data}")
        
        # Start measuring time
        start_time = datetime.now(timezone.utc)
        
        processed_data = preprocess_input(input_data, preprocessors)
        prediction = model.predict(processed_data)
        
        # Measure response time
        response_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        # Log metrics to BigQuery
        log_to_bigquery("/predict", input_data, prediction[0], response_time, "success")
        
        logger.info(f"Prediction: {prediction[0]}, Response time: {response_time} seconds")
        return jsonify({"prediction": int(prediction[0])})
    except Exception as e:
        error_message = f"An error occurred: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_message)  # Log the error on the server side
        
        # Log error metrics to BigQuery
        log_to_bigquery("/predict", input_data, None, 0, f"error: {str(e)}")
        
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
        logger.info("Model and preprocessors loaded successfully.")
        app.run(host="0.0.0.0", port=8000, debug=True)
    except Exception as e:
        logger.error(f"Error initializing the application: {e}")
