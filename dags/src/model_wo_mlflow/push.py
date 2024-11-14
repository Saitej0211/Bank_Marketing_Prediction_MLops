import os
import logging
from google.cloud import storage

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_FILE_PATH = os.path.join(PROJECT_DIR, "dags", "logs", "push_to_gcp.log")

# Key path for Google Cloud credentials (if needed)
KEY_PATH = "/opt/airflow/config/Key.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = KEY_PATH

# Create necessary directories for logging
os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)

# Custom file logger setup
file_handler = logging.FileHandler(LOG_FILE_PATH, mode='a')
logger.addHandler(file_handler)

def upload_to_gcs(local_path, bucket_name, destination_blob_name):
    """Upload a local directory to Google Cloud Storage."""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        # Check if bucket exists
        if not bucket.exists():
            raise ValueError(f"Bucket {bucket_name} does not exist.")

        for root, _, files in os.walk(local_path):
            for file in files:
                file_path = os.path.join(root, file)
                blob_path = os.path.join(destination_blob_name, os.path.relpath(file_path, local_path))
                blob = bucket.blob(blob_path)
                blob.upload_from_filename(file_path)
                logger.info(f"Uploaded {file_path} to gs://{bucket_name}/{blob_path}")
    except Exception as e:
        logger.error(f"Failed to upload model to GCS: {e}")
        raise

def push_to_gcp():
    logger.info("Starting the model upload process.")
    
    # Configuration variables (local path and GCS bucket details)
    local_model_path = "/tmp/model"  # Path where your local model is stored
    bucket_name = "mlopsprojectdatabucketgrp6"  
    destination_blob_name = "models/best_random_forest_model"

    # Step 1: Upload model to GCS
    try:
        upload_to_gcs(local_model_path, bucket_name, destination_blob_name)
    except Exception as e:
        logger.error(f"Failed to upload model to GCS: {e}")
        return  # Exit if the upload fails

    logger.info("Model upload process completed successfully.")

if __name__ == "__main__":
    push_to_gcp()