import os
from google.cloud import storage

# Specify your GCS bucket name and local model path
PAR_DIRECTORY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BUCKET_NAME = "mlopsprojectdatabucketgrp6"  # Replace with your bucket name
MODEL_PATH = os.path.join(PAR_DIRECTORY, "models", "random_forest_20241106-115427.pkl")  # Replace with your model path
MODEL_NAME = "random_forest_model"  # The base name for your model
SERVICE_ACCOUNT_KEY_PATH = os.path.join(PAR_DIRECTORY, "config", "Key.json") 

def set_google_credentials(service_account_key_path):
    """Set the GOOGLE_APPLICATION_CREDENTIALS environment variable."""
    if os.path.exists(service_account_key_path):
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_account_key_path
        print(f"Set GOOGLE_APPLICATION_CREDENTIALS to: {service_account_key_path}")
    else:
        raise FileNotFoundError(f"The specified service account key file does not exist at {service_account_key_path}")

def get_next_version(bucket_name, model_name):
    """Get the next version number for the model."""
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name, prefix=f"models/{model_name}_v")
    
    # Extract existing versions and find the highest one
    versions = []
    for blob in blobs:
        if blob.name.startswith(f"models/{model_name}_v"):
            version_str = blob.name.split('_v')[-1].split('.')[0]  # Extract version number from filename
            try:
                versions.append(int(version_str))
            except ValueError:
                continue
    
    # Return the next version number
    return max(versions, default=0) + 1

def upload_model_to_gcs(bucket_name, model_path, model_name):
    """Uploads the model file to a specified GCS bucket with dynamic versioning."""
    # Get the next version number
    version = get_next_version(bucket_name, model_name)
    
    # Create a GCS client
    storage_client = storage.Client()
    
    # Define the destination path in GCS
    destination_blob_name = f"models/{model_name}_v{version}.pkl"
    
    # Get the bucket
    bucket = storage_client.bucket(bucket_name)
    
    # Create a blob object from the file path
    blob = bucket.blob(destination_blob_name)
    
    # Upload the file to GCS
    blob.upload_from_filename(model_path)
    
    print(f"Model uploaded to gs://{bucket_name}/{destination_blob_name}")

def list_buckets():
    """List all buckets in GCP."""
    client = storage.Client()
    buckets = client.list_buckets()
    print("Buckets in GCP:")
    for bucket in buckets:
        print(bucket.name)

if __name__ == "__main__":
    # Path to your service account key
    
    # Set Google Cloud credentials
    set_google_credentials(SERVICE_ACCOUNT_KEY_PATH)

    # Upload the model to GCS
    upload_model_to_gcs(BUCKET_NAME, MODEL_PATH, MODEL_NAME)

    # List buckets to verify connection and permissions
    list_buckets()