import os
import pandas as pd
import logging
import tensorflow_data_validation as tfdv
from airflow.utils.log.logging_mixin import LoggingMixin
import pickle

# Define paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = os.path.join(PROJECT_DIR, "data", "processed")
LOG_DIR = os.path.join(PROJECT_DIR, "dags", "logs")

# Input and output file paths
INPUT_FILE_PATH = os.path.join(DATA_DIR, "raw_data.csv")
OUTPUT_PICKLE_PATH = os.path.join(DATA_DIR, "validate.pkl")
OUTPUT_CSV_PATH = os.path.join(DATA_DIR, "validate.csv")
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
    try:
        custom_log("Starting data loading process")

        # Check if the file exists
        if os.path.exists(INPUT_FILE_PATH):
            # Load data based on file extension
            if INPUT_FILE_PATH.endswith('.csv'):
                try:
                    df = pd.read_csv(INPUT_FILE_PATH, encoding='ISO-8859-1')  
                    custom_log(f"Loaded data from {INPUT_FILE_PATH} with shape {df.shape}")
                except Exception as e:
                    logging.error(f"Failed to read CSV file: {e}")
                    raise
            elif INPUT_FILE_PATH.endswith('.pkl'):
                try:
                    df = pd.read_pickle(INPUT_FILE_PATH)
                    custom_log(f"Loaded data from {INPUT_FILE_PATH} with shape {df.shape}")
                except Exception as e:
                    logging.error(f"Failed to read pickle file: {e}")
                    raise
            else:
                logging.error(f"Unsupported file type: {INPUT_FILE_PATH}")
                raise ValueError(f"Unsupported file type: {INPUT_FILE_PATH}")

            # Return the loaded DataFrame
            return df
        else:
            logging.error(f"File {INPUT_FILE_PATH} not found.")
            raise FileNotFoundError(f"{INPUT_FILE_PATH} not found.")

    except Exception as e:
        logging.error(f"Error in read_data: {e}")
        raise  # Re-raise the exception for upstream handling

def prepare_train_data(df):
    """Prepare training, evaluation, and serving data from the full dataset."""
    total_len = len(df)
    train_len = int(total_len * 0.6)
    eval_len = int(total_len * 0.2)

    train_df = df.iloc[:train_len].reset_index(drop=True)
    eval_df = df.iloc[train_len:train_len + eval_len].reset_index(drop=True)
    serving_df = df.iloc[train_len + eval_len:].drop(columns='y').reset_index(drop=True)

    custom_log(f"Prepared training data with shape {train_df.shape}")
    custom_log(f"Prepared evaluation data with shape {eval_df.shape}")
    custom_log(f"Prepared serving data with shape {serving_df.shape}")

    return train_df, eval_df, serving_df

def generate_train_stats(train_df):
    """Generate statistics from the training dataset."""
    train_stats = tfdv.generate_statistics_from_dataframe(train_df)
    custom_log("Generated training data statistics.")
    return train_stats

def generate_serving_stats(serving_df):
    """Generate statistics from the serving dataset."""
    serving_stats = tfdv.generate_statistics_from_dataframe(serving_df)
    custom_log("Generated serving data statistics.")
    return serving_stats

def infer_schema(train_stats):
    """Infer schema from the computed statistics."""
    schema = tfdv.infer_schema(statistics=train_stats)
    custom_log("Inferred schema from training data statistics.")
    return schema

def save_schema(schema, output_dir, suffix=''):
    """Save the schema to a file."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    schema_file = os.path.join(output_dir, f'schema{suffix}.pbtxt')
    tfdv.write_schema_text(schema, schema_file)
    custom_log(f"Schema saved to {schema_file}")
    return schema_file

def visualize_statistics(lhs_stats, rhs_stats, lhs_name="TRAIN_DATASET", rhs_name="EVAL_DATASET"):
    """Visualize statistics for comparison between two datasets."""
    tfdv.visualize_statistics(lhs_statistics=lhs_stats, rhs_statistics=rhs_stats, lhs_name=lhs_name, rhs_name=rhs_name)
    custom_log("Statistics visualization complete.")
    return "Statistics visualization complete."

def save_to_pickle(obj, file_path):
    """Save an object to a pickle file."""
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)
    custom_log(f"Saved object to pickle file at {file_path}")

def save_to_csv(df, file_path):
    """Save DataFrame to a CSV file."""
    df.to_csv(file_path, index=False)
    custom_log(f"DataFrame saved to CSV file at {file_path}")

def validate_data_schema():
    """Main function to run the end-to-end schema generation and validation process."""
    # Step 1: Load data
    df = read_data(INPUT_FILE_PATH)

    # Step 2: Prepare training, evaluation, and serving data
    train_df, eval_df, serving_df = prepare_train_data(df)

    # Step 3: Generate statistics for training data
    train_stats = generate_train_stats(train_df)

    # Step 4: Infer schema from training statistics
    schema = infer_schema(train_stats)

    # Step 5: Save schema
    schema_file = save_schema(schema, DATA_DIR)
    print(f"Training schema saved to: {schema_file}")
    print(f"Number of features in training schema: {len(schema.feature)}")

    # Step 6: Generate statistics for serving data
    serving_stats = generate_serving_stats(serving_df)

    # Step 7: Generate statistics for evaluation data
    eval_stats = generate_train_stats(eval_df)

    # Step 8: Visualize statistics for training vs. evaluation data
    visualization_result = visualize_statistics(lhs_stats=train_stats, rhs_stats=eval_stats)
    print(visualization_result)

    # Step 9: Save the complete dataset as 'validate_data.pkl'
    save_to_pickle(df, OUTPUT_PICKLE_PATH)
    print(f"Complete dataset saved as pickle file at {OUTPUT_PICKLE_PATH}")

    # Step 10: Save processed DataFrame as CSV
    save_to_csv(df, OUTPUT_CSV_PATH)
    print(f"Processed DataFrame saved as CSV file at {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    validate_data_schema()
