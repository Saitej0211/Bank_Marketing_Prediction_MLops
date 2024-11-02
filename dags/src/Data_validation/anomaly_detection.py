import os
import pandas as pd
import logging
import tensorflow as tf
import tensorflow_data_validation as tfdv
from tensorflow_data_validation.utils import slicing_util
from tensorflow_metadata.proto.v0.statistics_pb2 import DatasetFeatureStatisticsList
from tensorflow_metadata.proto.v0.schema_pb2 import Schema  # Import Schema directly
from src.Data_validation.data_schema_statistics_generation import (
    infer_schema, save_schema, validate_data_schema, read_data, prepare_train_data
)
from airflow.utils.log.logging_mixin import LoggingMixin

# Define paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = os.path.join(PROJECT_DIR, "data", "processed")
LOG_DIR = os.path.join(PROJECT_DIR, "dags", "logs")

# Input and output file paths
INPUT_FILE_PATH = os.path.join(DATA_DIR, "validate.pkl")
OUTPUT_PICKLE_PATH = os.path.join(DATA_DIR, "validate_process.pkl")
OUTPUT_CSV_PATH = os.path.join(DATA_DIR, "validate_process.csv") 
LOG_FILE_PATH = os.path.join(LOG_DIR, "anomaly_detection.log")

# Ensure necessary directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Set up Airflow logger
airflow_logger = LoggingMixin().log

# Set up custom file logger
file_logger = logging.getLogger('file_logger')
file_logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(LOG_FILE_PATH, mode='a')
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
file_logger.addHandler(file_handler)

def custom_log(message, level=logging.INFO):
    """Log to both Airflow and custom file logger."""
    if level == logging.INFO:
        airflow_logger.info(message)
        file_logger.info(message)
    elif level == logging.ERROR:
        airflow_logger.error(message)
        file_logger.error(message)
    elif level == logging.WARNING:
        airflow_logger.warning(message)
        file_logger.warning(message)

def read_data(input_file_path):
    """Read a CSV or pickle file and return it as a DataFrame."""
    try:
        custom_log("Starting data loading process")

        # Check if the file exists
        if os.path.exists(input_file_path):
            # Load data based on file extension
            if input_file_path.endswith('.csv'):
                try:
                    df = pd.read_csv(input_file_path, encoding='ISO-8859-1')  
                    custom_log(f"Loaded data from {input_file_path} with shape {df.shape}")
                except Exception as e:
                    custom_log(f"Failed to read CSV file: {e}", level=logging.ERROR)
                    raise
            elif input_file_path.endswith('.pkl'):
                try:
                    df = pd.read_pickle(input_file_path)
                    custom_log(f"Loaded data from {input_file_path} with shape {df.shape}")
                except Exception as e:
                    custom_log(f"Failed to read pickle file: {e}", level=logging.ERROR)
                    raise
            else:
                custom_log(f"Unsupported file type: {input_file_path}", level=logging.ERROR)
                raise ValueError(f"Unsupported file type: {input_file_path}")

            return df
        else:
            custom_log(f"File {input_file_path} not found.", level=logging.ERROR)
            raise FileNotFoundError(f"{input_file_path} not found.")

    except Exception as e:
        custom_log(f"Error in read_data: {e}", level=logging.ERROR)
        raise  # Re-raise the exception for upstream handling

def save_dataframe(df, output_pickle_path, output_csv_path):
    """Save a DataFrame as both a pickle and CSV file."""
    try:
        df.to_pickle(output_pickle_path)
        custom_log(f"Data successfully saved to {output_pickle_path}")

        df.to_csv(output_csv_path, index=False)
        custom_log(f"Data successfully saved to {output_csv_path}")

    except Exception as e:
        custom_log(f"Failed to save DataFrame: {e}", level=logging.ERROR)
        raise

def prepare_data_splits(df):
    """Prepare training, evaluation, and serving data splits."""
    train_len = int(len(df) * 0.7)
    eval_serv_len = len(df) - train_len
    eval_len = eval_serv_len // 2
    serv_len = eval_serv_len - eval_len

    train_df = df.iloc[:train_len].reset_index(drop=True)
    eval_df = df.iloc[train_len: train_len + eval_len].reset_index(drop=True)
    serving_df = df.iloc[train_len + eval_len: train_len + eval_len + serv_len].reset_index(drop=True)

    if 'y' in serving_df.columns:
        serving_df = serving_df.drop(columns=['y'])

    custom_log(f"Prepared data splits: train shape {train_df.shape}, eval shape {eval_df.shape}, serving shape {serving_df.shape}")
    return train_df, eval_df, serving_df

def calculate_and_display_anomalies(statistics, schema):
    """Calculate and display anomalies in the statistics."""
    anomalies = tfdv.validate_statistics(schema=schema, statistics=statistics)
    tfdv.display_anomalies(anomalies=anomalies)
    custom_log("Anomalies calculated and displayed.")

def visualize_slices_in_groups(sliced_stats, group_size=2):
    """Visualize slices of statistics in groups."""
    num_slices = len(sliced_stats.datasets)
    if num_slices == 0:
        custom_log("No slices available in the sliced statistics.", level=logging.WARNING)
        return

    for i in range(0, num_slices, group_size):
        stats_list = []
        names_list = []
        for j in range(i, min(i + group_size, num_slices)):
            slice_stats_list = DatasetFeatureStatisticsList()
            slice_stats_list.datasets.extend([sliced_stats.datasets[j]])
            stats_list.append(slice_stats_list)
            names_list.append(sliced_stats.datasets[j].name)

        if len(stats_list) > 1:
            for k in range(1, len(stats_list)):
                tfdv.visualize_statistics(
                    lhs_statistics=stats_list[k-1],
                    rhs_statistics=stats_list[k],
                    lhs_name=names_list[k-1],
                    rhs_name=names_list[k]
                )
        else:
            tfdv.visualize_statistics(
                lhs_statistics=stats_list[0],
                lhs_name=names_list[0]
            )
    custom_log("Sliced statistics visualized.")

def validate_data(input_file_path, output_pickle_path, output_csv_path):
    """Validate and process the data."""
    # Read the data from the file
    df = read_data(input_file_path)

    # Save the schema and validate the schema type
    train_stats, schema, schema_file = save_schema(input_file_path, os.path.dirname(output_pickle_path))

    custom_log(f"Schema type: {type(schema)}")  # Log the schema type

    train_df, eval_df, serving_df = prepare_data_splits(df)

    eval_stats = tfdv.generate_statistics_from_dataframe(eval_df)
    calculate_and_display_anomalies(eval_stats, schema=schema)

    options = tfdv.StatsOptions(schema=schema, infer_type_from_schema=True)
    serving_stats = tfdv.generate_statistics_from_dataframe(serving_df, stats_options=options)
    calculate_and_display_anomalies(serving_stats, schema=schema)

    schema.default_environment.extend(['TRAINING', 'SERVING'])
    tfdv.get_feature(schema, 'y').not_in_environment.append('SERVING')

    duration = tfdv.get_feature(schema, 'duration')
    duration.skew_comparator.infinity_norm.threshold = 0.03

    skew_drift_anomalies = tfdv.validate_statistics(train_stats, schema,
                                                    previous_statistics=eval_stats,
                                                    serving_statistics=serving_stats)
    tfdv.display_anomalies(skew_drift_anomalies)

    slice_fn = slicing_util.get_feature_value_slicer(features={'job': None, 'marital': None, 'education': None})
    slice_stats_options = tfdv.StatsOptions(schema=schema,
                                            experimental_slice_functions=[slice_fn],
                                            infer_type_from_schema=True)
    sliced_stats = tfdv.generate_statistics_from_dataframe(df, stats_options=slice_stats_options)
    visualize_slices_in_groups(sliced_stats, group_size=3)

    # Save the processed DataFrame as both a pickle and CSV file
    save_dataframe(df, output_pickle_path, output_csv_path)
    custom_log(f"Data processing complete and saved to {output_pickle_path} and {output_csv_path}")

def anomaly_detection():
    """Main function to run the data processing."""
    custom_log("Starting data processing...")
    try:
        # Process the data
        validate_data(INPUT_FILE_PATH, OUTPUT_PICKLE_PATH, OUTPUT_CSV_PATH) 
        custom_log("Data processing completed successfully.")
    except Exception as e:
        custom_log(f"An error occurred during data processing: {e}", level=logging.ERROR)

if __name__ == "__main__":
    anomaly_detection()
