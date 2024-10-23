import os
import json
from google.cloud import storage
from google.auth.exceptions import RefreshError
import logging
import pickle


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#DIRECTORY TO STORE THE DATA
DATA_DIR = os.path.join(PROJECT_DIR, "data", "processed")

#DIRECTORY TO GET THE KEY FROM TO ACCESS THE SERVICE ACCOUNT
KEY_PATH = os.path.join(PROJECT_DIR, "config", "Key.json")

def download_data_from_gcp(bucket_name):
    try:
        
        logs_dir = os.path.join(PROJECT_DIR, "logs")
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        
        #CREATE THE LOGGING DIRECTORY IF NOT CREATED
        log_file_path = os.path.join(logs_dir, 'download_data.log')
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            handlers=[logging.FileHandler(log_file_path, mode='w'),
                                      logging.StreamHandler()])
        
        logging.info("Project directory fetched successfully")
         
        # SET ENVIRONMENT VARIABLES FOR AUTHENTICATION
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = KEY_PATH

        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        
        # Google cloud set up
        try:
            storage_client = storage.Client()
        except RefreshError as e:
            logging.error(f"Error initializing storage client: {e}")
            return None

        bucket = storage_client.bucket(bucket_name)

        # Get the latest blob from the bucket
        blobs = list(bucket.list_blobs())
        if not blobs:
            logging.error(f"No files found in bucket {bucket_name}")
            return None
        latest_blob = max(blobs, key=lambda x: x.updated)
        
        # Download the latest file content
        file_content = latest_blob.download_as_string()
        logging.info(f"Latest file {latest_blob.name} downloaded from GCS.")
        
        # Pickle the file content
        pickled_file_path = os.path.join(DATA_DIR, "raw_data.pkl")
        with open(pickled_file_path, 'wb') as f:
            pickle.dump(file_content, f)
        
        logging.info(f"File content pickled and saved as {pickled_file_path}.")
        
        return pickled_file_path
    
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return None
    
#bucket_name = "mlopsprojectdatabucketgrp6"
#
#pickled_file_path = download_data_from_gcp(bucket_name)
#
## Verify the pickled file
#if pickled_file_path is not None:
#    if os.path.exists(pickled_file_path):
#        print(f"Pickled file exists at: {pickled_file_path}")
#        with open(pickled_file_path, 'rb') as f:
#            loaded_data = pickle.load(f)
#        print(f"Loaded data type: {type(loaded_data)}")
#        print(f"Loaded data size: {len(loaded_data)} bytes")
#    else:
#        print("Pickled file was not created.")
#else:
#    print("Failed to download and pickle the file.")