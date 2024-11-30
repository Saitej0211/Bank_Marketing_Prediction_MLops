from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from google.cloud import storage, bigquery, monitoring_v3
from datetime import datetime, timezone
import os
import warnings
import logging
from flask_cors import CORS
import traceback
from google.protobuf.timestamp_pb2 import Timestamp

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Define global constants
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data", "processed")
BIGQUERY_TABLE_ID = "dvc-lab-439300.model_metrics_dataset.metrics_log"
BUCKET_NAME = "mlopsprojectdatabucketgrp6"
MODEL_PATH = "models/best_random_forest_model/model.pkl"
PROJECT_ID = "dvc-lab-439300" 

# Global variables for the model and preprocessors
model = None
preprocessors = {}

def get_bigquery_client():
    """Create and return a BigQuery client with proper authentication using ADC."""
    return bigquery.Client()

def log_to_bigquery(endpoint, input_data, prediction, response_time, status):
    """Log metrics to BigQuery."""
    try:
        client = get_bigquery_client()
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
    except Exception as e:
        logger.error(f"Error logging to BigQuery: {str(e)}")
        logger.error(traceback.format_exc())

def log_to_cloud_monitoring(metric_type, value):
    """Send custom metrics to Cloud Monitoring."""
    try:
        client = monitoring_v3.MetricServiceClient()
        project_name = f"projects/{PROJECT_ID}"

        # Create a metric descriptor (only needs to be done once, so check first)
        metric_descriptor = monitoring_v3.MetricDescriptor()
        metric_descriptor.type = metric_type
        metric_descriptor.labels.append(monitoring_v3.LabelDescriptor(key="endpoint"))

        # Check if the metric already exists (to avoid re-creating it each time)
        try:
            client.get_metric_descriptor(name=f"{project_name}/metricDescriptors/{metric_type}")
        except Exception as e:
            # If the metric doesn't exist, create it
            client.create_metric_descriptor(name=project_name, metric_descriptor=metric_descriptor)
            logger.info(f"Created custom metric: {metric_type}")

        # Create the time series data
        time_series = monitoring_v3.TimeSeries()
        time_series.metric.type = metric_type
        time_series.resource.type = "global"

        # Add timestamp and value to the time series
        point = time_series.points.add()
        timestamp = Timestamp()
        timestamp.GetCurrentTime()  # Current timestamp

        # Set the value of the metric point
        point.interval.start_time = timestamp
        point.value.double_value = value  # For response_time, it might be a float

        # Send the time series to Cloud Monitoring
        client.create_time_series(name=project_name, time_series=[time_series])
        logger.info(f"Metric {metric_type} sent successfully.")
    except Exception as e:
        logger.error(f"Error sending {metric_type} to Cloud Monitoring: {e}")

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
            logger.info(f"Loaded encoder for column: {col}")

        with open(f"{data_dir}/scaler.pkl", "rb") as f:
            preprocessors["scaler"] = pickle.load(f)
        logger.info("Loaded scaler")

        with open(f"{data_dir}/normalizer.pkl", "rb") as f:
            preprocessors["normalizer"] = pickle.load(f)
        logger.info("Loaded normalizer")
    except Exception as e:
        logger.error(f"Error loading preprocessing objects: {str(e)}")
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
        logger.error(f"Error during preprocessing: {str(e)}")
        raise

def load_model_from_gcp():
    """Load the model from a GCP bucket using Application Default Credentials."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(MODEL_PATH)
        model_bytes = blob.download_as_bytes()
        logger.info("Model successfully loaded from GCP.")
        return pickle.loads(model_bytes)
    except Exception as e:
        logger.error(f"Error loading model from GCP: {str(e)}")
        raise

@app.route("/", methods=["GET"])
def index():
    """Serve the HTML page."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Handle HTTP POST requests for predictions."""
    try:
        input_data = request.json
        logger.info(f"Received input data: {input_data}")

        # Start measuring time
        start_time = datetime.now(timezone.utc)

        # Preprocess input and make prediction
        processed_data = preprocess_input(input_data, preprocessors)
        prediction = model.predict(processed_data)

        # Measure response time
        response_time = (datetime.now(timezone.utc) - start_time).total_seconds()

        # Log metrics to BigQuery
        log_to_bigquery("/predict", input_data, prediction[0], response_time, "success")

        # Send custom metrics to Cloud Monitoring
        log_to_cloud_monitoring("custom.googleapis.com/response_time", response_time)
        log_to_cloud_monitoring("custom.googleapis.com/prediction_status", 1 if prediction[0] == 1 else 0)

        logger.info(f"Prediction: {prediction[0]}, Response time: {response_time} seconds")
        return jsonify({"prediction": int(prediction[0])})
    except Exception as e:
        error_message = f"An error occurred: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_message)

        # Log error metrics to BigQuery
        log_to_bigquery("/predict", input_data, None, 0, f"error: {str(e)}")

        # Send custom metrics to Cloud Monitoring
        log_to_cloud_monitoring("custom.googleapis.com/response_time", 0)  # Log a 0 for failed requests
        log_to_cloud_monitoring("custom.googleapis.com/prediction_status", 0)

        return jsonify({"error": error_message}), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint to verify the service is running."""
    try:
        if model is None or not preprocessors:
            return jsonify({"status": "unhealthy", "details": "Model or preprocessors not loaded"}), 500
        return jsonify({"status": "healthy", "details": "Service is running"}), 200
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({"status": "unhealthy", "details": str(e)}), 500

if __name__ == "__main__":
    try:
        # Lazy load preprocessors and model
        preprocessors = load_preprocessing_objects(DATA_DIR)
        model = load_model_from_gcp()
        logger.info("Model and preprocessors loaded successfully.")
        app.run(host="0.0.0.0", port=8000, debug=True)
    except Exception as e:
        logger.error(f"Error initializing the application: {e}")
