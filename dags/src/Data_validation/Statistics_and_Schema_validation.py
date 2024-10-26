import os
import pickle
import logging
from google.cloud import storage
from google.auth.exceptions import RefreshError
from airflow.utils.log.logging_mixin import LoggingMixin
import tensorflow as tf
import tensorflow_data_validation as tfdv
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow_metadata.proto.v0 import schema_pb2

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

def load_and_split_data(file_path):
    """Load dataset from file and split into training and evaluation sets."""
    try:
        df = pd.read_csv(file_path, skipinitialspace=True)
        train_df, eval_df = train_test_split(df, test_size=0.2, shuffle=False)
        custom_log("Data successfully loaded and split into training and evaluation sets.")
        return train_df, eval_df
    except Exception as e:
        custom_log(f"Failed to load and split data: {e}", level=logging.ERROR)
        return None, None

def generate_statistics(df, dataset_name=""):
    """Generate statistics for a given dataset using TFDV."""
    try:
        stats = tfdv.generate_statistics_from_dataframe(df)
        custom_log(f"Statistics generated for {dataset_name}.")
        return stats
    except Exception as e:
        custom_log(f"Failed to generate statistics for {dataset_name}: {e}", level=logging.ERROR)
        return None

def infer_and_update_schema(train_stats):
    """Infer schema from training stats and update it with domain constraints."""
    try:
        schema = tfdv.infer_schema(statistics=train_stats)
        valid_months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        valid_days_of_week = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
        
        for feature in schema.feature:
            if feature.name == 'age':
                feature.int_domain.min = 18
                feature.int_domain.max = 100
                feature.value_count.min = 1
                feature.value_count.max = 1
            elif feature.name == 'month':
                feature.ClearField('domain')
                feature.string_domain.value[:] = valid_months
                feature.value_count.min = 1
                feature.value_count.max = 1
            elif feature.name == 'day_of_week':
                feature.ClearField('domain')
                feature.string_domain.value[:] = valid_days_of_week
                feature.value_count.min = 1
                feature.value_count.max = 1

        custom_log("Schema inferred and updated with constraints.")
        return schema
    except Exception as e:
        custom_log(f"Failed to infer and update schema: {e}", level=logging.ERROR)
        return None

def visualize_statistics(lhs_stats, rhs_stats, lhs_name="TRAIN_DATASET", rhs_name="EVAL_DATASET"):
    """Visualize statistics for comparison between two datasets."""
    try:
        tfdv.visualize_statistics(lhs_statistics=lhs_stats, rhs_statistics=rhs_stats, lhs_name=lhs_name, rhs_name=rhs_name)
        custom_log("Statistics visualization complete.")
    except Exception as e:
        custom_log(f"Failed to visualize statistics: {e}", level=logging.ERROR)

# Example usage (for testing purposes)
if __name__ == "__main__":
    train_df, eval_df = load_and_split_data(os.path.join(DATA_DIR, "raw_data.csv"))
    if train_df is not None and eval_df is not None:
        train_stats = generate_statistics(train_df, "Training Set")
        eval_stats = generate_statistics(eval_df, "Evaluation Set")
        schema = infer_and_update_schema(train_stats)
        visualize_statistics(lhs_stats=eval_stats, rhs_stats=train_stats)
