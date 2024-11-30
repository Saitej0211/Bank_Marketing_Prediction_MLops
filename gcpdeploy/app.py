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

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Define global constants
PROJECT_ID = "dvc-lab-439300"  # Your Google Cloud Project ID
BIGQUERY_TABLE_ID = f"{PROJECT_ID}.model_metrics_dataset.metrics_log"
BUCKET_NAME = "mlopsprojectdatabucketgrp6"
MODEL_PATH = "models/best_random_forest_model/model.pkl"
METRICS_PREFIX = "custom.googleapis.com"

# Global variables for the model and preprocessors
model = None
preprocessors = {}

# Initialize the Cloud Monitoring client and set up metric descriptors once
monitoring_client = monitoring_v3.MetricServiceClient()
project_name = f"projects/{PROJECT_ID}"
response_time_descriptor = None
prediction_status_descriptor = None

def create_metric_descriptors():
    global response_time_descriptor, prediction_status_descriptor

    # Metric descriptor for response time
    response_time_descriptor = monitoring_v3.MetricDescriptor(
        type=f"{METRICS_PREFIX}/response_time",
        labels=[monitoring_v3.LabelDescriptor(key="endpoint", value_type="STRING")],
        metric_kind=monitoring_v3.MetricDescriptor.MetricKind.GAUGE,
        value_type=monitoring_v3.MetricDescriptor.ValueType.DOUBLE,
        unit="s",
        description="Response time of the model prediction endpoint."
    )

    # Metric descriptor for prediction status
    prediction_status_descriptor = monitoring_v3.MetricDescriptor(
        type=f"{METRICS_PREFIX}/prediction_status",
        labels=[monitoring_v3.LabelDescriptor(key="endpoint", value_type="STRING")],
        metric_kind=monitoring_v3.MetricDescriptor.MetricKind.GAUGE,
        value_type=monitoring_v3.MetricDescriptor.ValueType.INT64,
        unit="1",
        description="Prediction status: 0 for error, 1 for success."
    )

    # Create the metric descriptors in Cloud Monitoring
    try:
        monitoring_client.create_metric_descriptor(name=project_name, metric_descriptor=response_time_descriptor)
        monitoring_client.create_metric_descriptor(name=project_name, metric_descriptor=prediction_status_descriptor)
        logger.info("Metric descriptors created successfully.")
    except Exception as e:
        logger.error(f"Error creating metric descriptors: {str(e)}")

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

def log_to_cloud_monitoring(endpoint, response_time, prediction_status):
    """Send metrics to Cloud Monitoring."""
    try:
        # Create the time series data for response_time
        series_response_time = monitoring_v3.TimeSeries()
        series_response_time.metric.type = response_time_descriptor.type
        series_response_time.resource.type = "global"
        series_response_time.metric.labels["endpoint"] = endpoint
        series_response_time.points.add(
            value=monitoring_v3.Point(value=monitoring_v3.TypedValue(double_value=response_time)),
            interval=monitoring_v3.TimeInterval(
                end_time=monitoring_v3.Timestamp(seconds=int(datetime.now(timezone.utc).timestamp()))
            )
        )

        # Create the time series data for prediction_status
        series_prediction_status = monitoring_v3.TimeSeries()
        series_prediction_status.metric.type = prediction_status_descriptor.type
        series_prediction_status.resource.type = "global"
        series_prediction_status.metric.labels["endpoint"] = endpoint
        series_prediction_status.points.add(
            value=monitoring_v3.Point(value=monitoring_v3.TypedValue(int64_value=prediction_status)),
            interval=monitoring_v3.TimeInterval(
                end_time=monitoring_v3.Timestamp(seconds=int(datetime.now(timezone.utc).timestamp()))
            )
        )

        # Write the time series data
        monitoring_client.create_time_series(name=project_name, time_series=[series_response_time, series_prediction_status])
        logger.info(f"Metrics sent to Cloud Monitoring for {endpoint}.")
    except Exception as e:
        logger.error(f"Error sending metrics to Cloud Monitoring: {str(e)}")
        logger.error(traceback.format_exc())

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
    input_data = None  # Initialize the variable to avoid referencing before assignment in error block
    try:
        input_data = request.json
        logger.info(f"Received input data: {input_data}")

        # Preprocess the input data
        processed_input = preprocess_input(input_data, preprocessors)

        # Get model prediction
        prediction = model.predict(processed_input)
        prediction_proba = model.predict_proba(processed_input)

        logger.info(f"Prediction: {prediction}, Prediction Probability: {prediction_proba}")

        # Log metrics
        start_time = datetime.now()
        log_to_bigquery("/predict", input_data, prediction, (datetime.now() - start_time).total_seconds(), 1)
        log_to_cloud_monitoring("/predict", (datetime.now() - start_time).total_seconds(), 1)

        return jsonify({"prediction": prediction.tolist(), "probability": prediction_proba.tolist()})
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        log_to_bigquery("/predict", input_data, None, 0, 0)
        log_to_cloud_monitoring("/predict", 0, 0)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Load preprocessing objects and model at startup
    preprocessors = load_preprocessing_objects("preprocessors")
    model = load_model_from_gcp()
    create_metric_descriptors()  # Ensure metrics descriptors are created before starting server
    app.run(debug=True, host="0.0.0.0", port=8080)
