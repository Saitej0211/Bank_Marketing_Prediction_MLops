import os
import logging
from google.cloud import storage
import pickle
from sklearn.metrics import accuracy_score
from datetime import datetime
import tempfile
import pandas as pd

# Set logging configurations
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Loading file paths for CSVs
PAR_DIRECTORY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BUCKET_NAME = "mlopsprojectdatabucketgrp6"
MODEL_DIR = os.path.join(PAR_DIRECTORY, 'models')
NEW_MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_20241106-121604.pkl")  # Update this path dynamically if needed

# Set Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(PAR_DIRECTORY, "config", "Key.json")

# Define the path for the test data
TEST_DATA_PATH = os.path.join(PAR_DIRECTORY, "data", "processed", "test_data.csv")

def load_model_from_gcs(bucket_name, blob_name):
    """Load model from GCS"""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            blob.download_to_filename(temp_file.name)
            logger.info(f"Downloaded model from GCS: {blob_name}")
            with open(temp_file.name, 'rb') as f:
                model = pickle.load(f)
        
        return model
    except Exception as e:
        logger.error(f"Error downloading model from GCS: {e}")
        raise

def save_model_locally(model, model_name):
    """Save the model to a local directory under 'models'."""
    try:
        model_path = os.path.join(MODEL_DIR, model_name)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        logger.info(f"Model saved locally to {model_path}")
    except Exception as e:
        logger.error(f"Error saving model locally: {e}")
        raise

def compare_models(new_model, old_model, X_test, y_test):
    """Compare new model with old model and return True if new model is better."""
    new_accuracy = accuracy_score(y_test, new_model.predict(X_test))
    old_accuracy = accuracy_score(y_test, old_model.predict(X_test))
    return new_accuracy > old_accuracy

def upload_model_to_gcs(model, bucket_name, version):
    """Upload the model to GCS with versioning"""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        # Create a versioned blob name
        model_version_name = f"models/model_v{version}.pkl"

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            with open(temp_file.name, 'wb') as f:
                pickle.dump(model, f)
            
            blob = bucket.blob(model_version_name)
            blob.upload_from_filename(temp_file.name)
        
        logger.info(f"Uploaded model to GCS: {model_version_name}")
    except Exception as e:
        logger.error(f"Error uploading model to GCS: {e}")
        raise

def get_latest_model_version():
    """Get the latest model version number from GCS"""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blobs = list(bucket.list_blobs(prefix="models/"))
        
        if not blobs:
            logger.info("No models found in GCS. Uploading new model.")
            return 0  # First model, no previous versions

        # Extract version numbers from blob names and return the highest version
        versions = []
        for blob in blobs:
            if "model_v" in blob.name:
                version = int(blob.name.split("model_v")[1].split(".")[0])
                versions.append(version)
        
        latest_version = max(versions) if versions else 0
        return latest_version
    except Exception as e:
        logger.error(f"Error retrieving the latest model version from GCS: {e}")
        raise

def load_test_data(test_path):
    """Load test data from a CSV file."""
    try:
        test_data = pd.read_csv(test_path)
        logger.info(f"Loaded test data from {test_path} with shape {test_data.shape}")
        
        X_test = test_data.drop('y', axis=1)  # Assuming 'y' is the target variable
        y_test = test_data['y']
        return X_test, y_test
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        raise

def main():
    """Main function to compare models and upload the better one with versioning"""
    try:
        # Load test data
        X_test, y_test = load_test_data(TEST_DATA_PATH)

        # Get the latest model version
        latest_version = get_latest_model_version()

        # Load the newly trained model
        with open(NEW_MODEL_PATH, 'rb') as f:
            new_model = pickle.load(f)

        # If it's the first model (version 0), upload it directly
        if latest_version == 0:
            upload_model_to_gcs(new_model, BUCKET_NAME, version=1)
            logger.info("First model uploaded to GCS with version 1.")
        else:
            # Load the old model from GCS
            old_model_blob_name = f"models/model_v{latest_version}.pkl"
            old_model = load_model_from_gcs(BUCKET_NAME, old_model_blob_name)

            # Compare the models and upload the better one
            if compare_models(new_model, old_model, X_test, y_test):
                new_version = latest_version + 1
                upload_model_to_gcs(new_model, BUCKET_NAME, version=new_version)
                logger.info(f"New model uploaded to GCS as version {new_version}, it outperforms the old model.")
            else:
                logger.info(f"Old model (v{latest_version}) is better. No need to upload the new model.")
    
    except Exception as e:
        logger.error(f"Error in the model comparison process: {e}")
        raise

if __name__ == "__main__":
    main()