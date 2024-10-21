import os
import json
from google.cloud import storage
import logging
import pickle

def load_data_from_gcp(**kwargs):
    try:
        PROJECT_DIR = os.getcwd()
        logging.info("Project directory fetched succesfully")
        data_dir = kwargs['data_dir']
        bucket_name = kwargs['bucket_name']
        KEY_PATH = kwargs['KEY_PATH']

        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = KEY_PATH

        destination_dir = os.path.join(PROJECT_DIR, "dags", "processed", "Fetched")

        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)
        
        #Google cloud set up here
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        #Get the latest blob from the bucket
        blobs = list(bucket.list_blobs())
        if not blobs:
            raise ValueError(f"No files found in bucket {bucket_name}")
        latest_blob = max(blobs, key=lambda x: x.updated)
        
        

        # Download the latest file content
        file_content = latest_blob.download_as_string()
        print(f"Latest file {latest_blob.name} downloaded from GCS.")
        logging.info(f"Latest file {latest_blob.name} downloaded from GCS.")
        
        # Pickle the file content
        pickled_file_path = os.path.join(destination_dir, "raw_data.pkl")
        with open(pickled_file_path, 'wb') as f:
            pickle.dump(file_content, f)
        
        print(f"File content pickled and saved as {pickled_file_path}.")
        logging.info(f"File content pickled and saved as {pickled_file_path}.")
        
        return pickled_file_path
        
    except Exception as e:
        print("An unexpected error occured: {e}")
        logging.error("An unexpected error occured: {e}")

