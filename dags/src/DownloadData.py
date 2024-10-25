import os
import pickle
import logging
from google.cloud import storage
from google.auth.exceptions import RefreshError
from airflow.utils.log.logging_mixin import LoggingMixin

# Set up Airflow logger
airflow_logger = LoggingMixin().log

# Set up custom file logger
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_DIR = os.path.join(PROJECT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOG_DIR, 'download_data.log')

file_logger = logging.getLogger('file_logger')
file_logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(LOG_FILE_PATH, mode='a')
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
file_logger.addHandler(file_handler)

# Set up project directories
DATA_DIR = os.path.join(PROJECT_DIR, "data", "processed")
os.makedirs(DATA_DIR, exist_ok=True)

def custom_log(message, level=logging.INFO):
    """Log to both Airflow and custom file logger"""
    if level == logging.INFO:
        airflow_logger.info(message)
        file_logger.info(message)
    elif level == logging.ERROR:
        airflow_logger.error(message)
        file_logger.error(message)
    elif level == logging.WARNING:
        airflow_logger.warning(message)
        file_logger.warning(message)

def download_data_from_gcp(bucket_name):
    try:
        custom_log("Starting data download process")
        
        # Set environment variables for authentication
        KEY_PATH = os.path.join(PROJECT_DIR, "config", "key.json")
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = KEY_PATH
        
        # Google cloud set up
        try:
            storage_client = storage.Client()
        except RefreshError as e:
            custom_log(f"Error initializing storage client: {e}", level=logging.ERROR)
            return None

        bucket = storage_client.bucket(bucket_name)

        # Get the latest blob from the bucket
        blobs = list(bucket.list_blobs())
        if not blobs:
            custom_log(f"No files found in bucket {bucket_name}", level=logging.ERROR)
            return None
        latest_blob = max(blobs, key=lambda x: x.updated)
        
        # Download the latest file content
        file_content = latest_blob.download_as_string()
        custom_log(f"Latest file {latest_blob.name} downloaded from GCS.")
        
        # Pickle the file content
        pickled_file_path = os.path.join(DATA_DIR, "raw_data.pkl")
        with open(pickled_file_path, 'wb') as f:
            pickle.dump(file_content, f)
        
        custom_log(f"File content pickled and saved as {pickled_file_path}.")
        
        return pickled_file_path
    
    except Exception as e:
        custom_log(f"An unexpected error occurred: {e}", level=logging.ERROR)
        return None

# Example usage (this won't run in Airflow, but can be used for local testing)
if __name__ == "__main__":
    bucket_name = "mlopsprojectdatabucketgrp6"
    result = download_data_from_gcp(bucket_name)
    if result:
        custom_log(f"Data successfully downloaded and saved to {result}")
    else:
        custom_log("Failed to download data from GCP", level=logging.ERROR)