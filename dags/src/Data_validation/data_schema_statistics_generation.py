# data_schema_statistics_generation.py
 
import os
import json
import pandas as pd
import numpy as np
from great_expectations.core import ExpectationSuite
from great_expectations.dataset import PandasDataset
from airflow.utils.log.logging_mixin import LoggingMixin
import logging
 
# Define paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = os.path.join(PROJECT_DIR, "data", "processed")
LOG_DIR = os.path.join(PROJECT_DIR, "dags", "logs")
 
# Input and output file paths
INPUT_FILE_PATH = os.path.join(DATA_DIR, "raw_data.csv")
OUTPUT_PICKLE_PATH = os.path.join(DATA_DIR, "validate.pkl")
LOG_FILE_PATH = os.path.join(LOG_DIR, "data_schema_statistics_generation.log")
 
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
 
def read_data(file_path):
    """Load the data from the CSV file and process it if needed."""
    data = pd.read_csv(file_path)
    custom_log(f"Loaded data from {file_path} with shape {data.shape}")
 
    # Check if the first column contains strings with commas
    if data.shape[1] > 0 and isinstance(data.iloc[0, 0], str) and ',' in data.iloc[0, 0]:
        column_names = data.columns[0].split(',')
        data = data.iloc[:, 0].str.split(',', expand=True)
        data.columns = column_names
 
    return data
 
def prepare_train_data(df):
    """Prepare training, evaluation, and serving data from the full dataset."""
    total_len = len(df)
    train_len = int(total_len * 0.6)
    eval_len = int(total_len * 0.2)

    train_df = df.iloc[:train_len].reset_index(drop=True)
    eval_df = df.iloc[train_len:train_len + eval_len].reset_index(drop=True)

    # Check if 'y' exists before dropping it
    if 'y' in df.columns:
        serving_df = df.iloc[train_len + eval_len:].drop(columns='y').reset_index(drop=True)
    else:
        serving_df = df.iloc[train_len + eval_len:].reset_index(drop=True)
        custom_log("Column 'y' not found, skipping drop for serving data.", level=logging.WARNING)

    custom_log(f"Prepared training data with shape {train_df.shape}")
    custom_log(f"Prepared evaluation data with shape {eval_df.shape}")
    custom_log(f"Prepared serving data with shape {serving_df.shape}")

    return train_df, eval_df, serving_df

 
def generate_statistics(df):
    """Generate statistics from the dataset."""
    ge_df = PandasDataset(df)
    suite = ExpectationSuite("my_suite")
    
    # Add some expectations (you can customize these based on your data)
    ge_df.expect_column_values_to_not_be_null("duration")
    ge_df.expect_column_values_to_be_between("age", min_value=18, max_value=100)
    
    # Add the expectations to the suite
    for expectation in ge_df.get_expectation_suite().expectations:
        suite.add_expectation(expectation)
    
    custom_log("Generated data statistics.")
    return suite
 
def save_schema(stats, output_dir, suffix=''):
    """Save the schema to a file."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    schema_file = os.path.join(output_dir, f'schema{suffix}.json')
    
    # Convert ExpectationSuite to a dictionary
    schema_dict = stats.to_json_dict()
    
    # Save the dictionary as a JSON file
    with open(schema_file, 'w') as f:
        json.dump(schema_dict, f, indent=2)
    
    custom_log(f"Schema saved to {schema_file}")
    return schema_file
 
def validate_data_schema():
    """Main function to run the end-to-end schema generation and validation process."""
    # Step 1: Load data
    df = read_data(INPUT_FILE_PATH)
 
    # Step 2: Prepare training, evaluation, and serving data
    train_df, eval_df, serving_df = prepare_train_data(df)
 
    # Step 3: Generate statistics for training data
    train_stats = generate_statistics(train_df)
 
    # Step 4: Save training schema to output directory
    schema_file = save_schema(train_stats, DATA_DIR, suffix='_training')
    print(f"Training schema saved to: {schema_file}")
 
    # Step 5: Generate statistics for serving data
    serving_stats = generate_statistics(serving_df)
 
    # Step 6: Save serving schema to output directory
    serving_schema_file = save_schema(serving_stats, DATA_DIR, suffix='_serving')
    print(f"Serving schema saved to: {serving_schema_file}")
 
    # Step 7: Generate statistics for evaluation data
    eval_stats = generate_statistics(eval_df)
 
    # Step 8: Compare statistics
    train_expectations = train_stats.expectations
    eval_expectations = eval_stats.expectations
    
    for train_exp, eval_exp in zip(train_expectations, eval_expectations):
        if train_exp.to_json_dict() != eval_exp.to_json_dict():
            print(f"Difference found in expectation: {train_exp.expectation_type}")
 
    # Save processed data
    df.to_pickle(OUTPUT_PICKLE_PATH)
    custom_log(f"Processed data saved to {OUTPUT_PICKLE_PATH}")
 
if __name__ == "__main__":
    validate_data_schema()

