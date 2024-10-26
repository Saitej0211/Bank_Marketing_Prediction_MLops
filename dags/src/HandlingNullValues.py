import os
import logging
import pandas as pd
import io
from airflow.utils.log.logging_mixin import LoggingMixin

# Define paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_DIR, "data", "processed")
INPUT_FILE_PATH = os.path.join(DATA_DIR, "raw_data.csv")
PICKLE_FILE_PATH = os.path.join(DATA_DIR, "processed_data.pkl")
INFO_CSV_PATH = os.path.join(DATA_DIR, "dataframe_info.csv")
DESCRIPTION_CSV_PATH = os.path.join(DATA_DIR, "dataframe_description.csv")
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
LOG_FILE_PATH = os.path.join(LOG_DIR, 'process_data.log')

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

def process_data(input_file_path=INPUT_FILE_PATH):
    """
    Process the input CSV data, perform data cleaning, and save results.
    """
    try:
        custom_log("Starting data processing")
        custom_log(f"Input file: {input_file_path}")

        # Load CSV data
        df = pd.read_csv(input_file_path, sep=';')
        custom_log(f"Data loaded successfully. DataFrame shape: {df.shape}")

        # Save DataFrame info
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue().strip().split('\n')
        info_df = pd.DataFrame(info_str[2:], columns=['Info'])
        info_df.to_csv(INFO_CSV_PATH, index=False)
        custom_log(f"DataFrame info saved to {INFO_CSV_PATH}")

        # Save DataFrame description
        description_df = df.describe()
        description_df.to_csv(DESCRIPTION_CSV_PATH)
        custom_log(f"DataFrame description saved to {DESCRIPTION_CSV_PATH}")

        # Check for and handle duplicate rows
        duplicate_rows = df.duplicated().sum()
        if duplicate_rows > 0:
            custom_log(f"Found {duplicate_rows} duplicate rows. Dropping duplicates.")
            df = df.drop_duplicates()
        else:
            custom_log("No duplicate rows found.")

        # Handle null values
        null_percentage = df.isnull().mean() * 100
        features_to_drop = null_percentage[null_percentage > 50].index.tolist()

        if features_to_drop:
            df_dropped = df.drop(columns=features_to_drop)
            custom_log(f"Dropped features with >50% null values: {features_to_drop}")
        else:
            df_dropped = df
            custom_log("No features dropped due to null values")

        # Save processed data as a pickle file
        df_dropped.to_pickle(PICKLE_FILE_PATH)
        custom_log(f"Processed data saved as pickle at {PICKLE_FILE_PATH}")
        custom_log("Data processing completed successfully")

        return PICKLE_FILE_PATH
    except FileNotFoundError:
        custom_log(f"Input file not found: {input_file_path}", level=logging.ERROR)
    except pd.errors.EmptyDataError:
        custom_log("The input CSV file is empty", level=logging.ERROR)
    except Exception as e:
        custom_log(f"An unexpected error occurred during data processing: {e}", level=logging.ERROR)

if __name__ == "__main__":
    process_data()