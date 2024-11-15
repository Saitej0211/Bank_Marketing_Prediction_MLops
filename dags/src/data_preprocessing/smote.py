import os
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, Normalizer
from airflow.utils.log.logging_mixin import LoggingMixin
import pickle
from google.cloud import storage
import logging
import io
from datetime import datetime
import time

# Set up Airflow logger
airflow_logger = LoggingMixin().log

# Set up file logger
file_logger = logging.getLogger('file_logger')
file_logger.setLevel(logging.INFO)

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
        
#key path
KEY_PATH = "/opt/airflow/config/Key.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = KEY_PATH

def upload_to_gcs(data, bucket_name, destination_blob_name, as_pickle=False):
    """Upload an in-memory buffer (CSV or Pickle) to Google Cloud Storage, 
    updating the file if it exists or creating it if it doesn't."""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        # Check if the bucket exists
        if not bucket.exists():
            raise ValueError(f"Bucket {bucket_name} does not exist.")

        blob = bucket.blob(destination_blob_name)

        # If the blob already exists, it will be overwritten by default
        if blob.exists():
            custom_log(f"File {destination_blob_name} already exists. Overwriting...")
        else:
            custom_log(f"File {destination_blob_name} does not exist. Creating new file...")

        # If we are uploading a pickle, use `upload_from_string` with pickle data
        if as_pickle:
            blob.upload_from_string(pickle.dumps(data), content_type="application/octet-stream")
            custom_log(f"Uploaded pickle data to gs://{bucket_name}/{destination_blob_name}")
        else:
            # For CSV or text data
            blob.upload_from_file(data, content_type="text/csv", rewind=True)
            custom_log(f"Uploaded CSV data to gs://{bucket_name}/{destination_blob_name}")
    except Exception as e:
        custom_log(f"Failed to upload to GCS: {e}", level=logging.ERROR)
        raise

def smote_analysis(input_file_path):
    try:
        custom_log("Starting SMOTE analysis with scaling and normalization")
        custom_log(f"Input file: {input_file_path}")
        bucket_name = "mlopsprojectdatabucketgrp6"

        # Load the encoded data
        df = pd.read_pickle(input_file_path)
        custom_log(f"Loaded data from {input_file_path} with shape {df.shape}")

        # Assuming the last column is the target variable
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        custom_log(f"Target variable distribution before processing:\n{y.value_counts(normalize=True)}")

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Apply Min-Max scaling
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        custom_log("Applied Min-Max scaling")

        # Apply normalization
        normalizer = Normalizer()
        X_train_normalized = normalizer.fit_transform(X_train_scaled)
        X_test_normalized = normalizer.transform(X_test_scaled)

        custom_log("Applied normalization")

        # Apply SMOTE
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_normalized, y_train)

        custom_log(f"Shape of training data before SMOTE: {X_train.shape}")
        custom_log(f"Shape of training data after SMOTE: {X_train_resampled.shape}")
        custom_log(f"Target variable distribution after SMOTE:\n{pd.Series(y_train_resampled).value_counts(normalize=True)}")

        # Combine the resampled training data
        df_train_resampled = pd.concat([pd.DataFrame(X_train_resampled, columns=X.columns), 
                                        pd.Series(y_train_resampled, name=y.name)], axis=1)

        # Combine the test data
        df_test = pd.DataFrame(X_test_normalized, columns=X.columns)
        df_test[y.name] = y_test.reset_index(drop=True)

        # Save the resampled training data as CSV and upload to GCS directly with a timestamp
        train_csv_buffer = io.StringIO()
        df_train_resampled.to_csv(train_csv_buffer, index=False)

        try:
            upload_to_gcs(train_csv_buffer, bucket_name, "files/md5/75/Processed_Files/smote_resampled_train_data")
            print("Training data uploaded successfully.")
        except Exception as e:
            print(f"Error uploading training data: {e}")
            time.sleep(2)  # Wait before retrying
            try:
                upload_to_gcs(train_csv_buffer, bucket_name, "files/md5/75/Processed_Files/smote_resampled_train_data")
                print("Training data uploaded successfully on retry.")
            except Exception as e:
                print(f"Final attempt to upload training data failed: {e}")

        # Save the test data as CSV and upload to GCS directly with a timestamp
        test_csv_buffer = io.StringIO()
        df_test.to_csv(test_csv_buffer, index=False)

        try:
            upload_to_gcs(test_csv_buffer, bucket_name, "files/md5/75/Processed_Files/test_data")
            print("Test data uploaded successfully.")
        except Exception as e:
            print(f"Error uploading test data: {e}")
            time.sleep(2)  # Wait before retrying
            try:
                upload_to_gcs(test_csv_buffer, bucket_name, "files/md5/75/Processed_Files/test_data")
                print("Test data uploaded successfully on retry.")
            except Exception as e:
                print(f"Final attempt to upload test data failed: {e}")

        custom_log("SMOTE analysis with scaling, normalization, and GCS upload completed successfully")

    except Exception as e:
        custom_log(f"An error occurred during SMOTE analysis: {e}", level=logging.ERROR)
        raise

if __name__ == "__main__":
    bucket_name = "your-gcs-bucket-name"  # Replace with your GCS bucket name
    input_file_path = os.path.join("data", "processed", "encoded_data.pkl")  # Adjust this path if needed
    smote_analysis(input_file_path, bucket_name)