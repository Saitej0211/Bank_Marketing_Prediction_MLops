import os
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, Normalizer
from airflow.utils.log.logging_mixin import LoggingMixin
import pickle

# Define paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = os.path.join(PROJECT_DIR, "data", "processed")
LOG_DIR = os.path.join(PROJECT_DIR, "dags", "logs")

# Ensure necessary directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Set up Airflow logger
airflow_logger = LoggingMixin().log

# Set up file logger
import logging
LOG_FILE_PATH = os.path.join(LOG_DIR, 'smote_analysis.log')
file_logger = logging.getLogger('file_logger')
file_logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(LOG_FILE_PATH, mode='a')
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
file_logger.addHandler(file_handler)

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

def smote_analysis(input_file_path):
    try:
        custom_log("Starting SMOTE analysis with scaling and normalization")
        custom_log(f"Input file: {input_file_path}")

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
        df_test = pd.concat([pd.DataFrame(X_test_normalized, columns=X.columns), 
                             pd.Series(y_test, name=y.name)], axis=1)

        # Save the resampled training data as CSV
        train_csv_path = os.path.join(DATA_DIR, "smote_resampled_train_data.csv")
        df_train_resampled.to_csv(train_csv_path, index=False)
        custom_log(f"Saved SMOTE resampled training data to CSV: {train_csv_path}")

        # Save the test data as CSV
        test_csv_path = os.path.join(DATA_DIR, "test_data.csv")
        df_test.to_csv(test_csv_path, index=False)
        custom_log(f"Saved test data to CSV: {test_csv_path}")

        # Save the resampled training data as pickle
        train_pkl_path = os.path.join(DATA_DIR, "smote_resampled_train_data.pkl")
        with open(train_pkl_path, 'wb') as f:
            pickle.dump({
                'X_train': X_train_resampled, 
                'y_train': y_train_resampled,
                'scaler': scaler,
                'normalizer': normalizer
            }, f)
        custom_log(f"Saved SMOTE resampled training data to pickle: {train_pkl_path}")

        # Save the test data as pickle
        test_pkl_path = os.path.join(DATA_DIR, "test_data.pkl")
        with open(test_pkl_path, 'wb') as f:
            pickle.dump({
                'X_test': X_test_normalized, 
                'y_test': y_test,
                'scaler': scaler,
                'normalizer': normalizer
            }, f)
        custom_log(f"Saved test data to pickle: {test_pkl_path}")

        custom_log("SMOTE analysis with scaling and normalization completed successfully")
        return train_pkl_path, test_pkl_path

    except Exception as e:
        custom_log(f"An error occurred during SMOTE analysis: {e}", level=logging.ERROR)
        raise

if __name__ == "__main__":
    input_file_path = os.path.join(DATA_DIR, "encoded_data.pkl")  # Adjust this path if needed
    train_path, test_path = smote_analysis(input_file_path)
    print(f"Resampled training data saved to: {train_path}")
    print(f"Test data saved to: {test_path}")