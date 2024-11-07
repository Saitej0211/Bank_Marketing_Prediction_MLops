import os
import io
import pandas as pd
import pickle
import logging
from airflow.utils.log.logging_mixin import LoggingMixin

# Set up project directories and file paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_DIR, "data", "processed")
OUTPUT_FILE_PATH = os.path.join(DATA_DIR, "raw_data.csv")
PICKLE_FILE_PATH = os.path.join(DATA_DIR, "raw_data.pkl")
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
LOG_FILE_PATH = os.path.join(LOG_DIR, 'load_data.log')

# Ensure necessary directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Set up custom file logger
file_logger = logging.getLogger('file_logger')
file_logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(LOG_FILE_PATH, mode='a')
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
file_logger.addHandler(file_handler)

# Set up Airflow logger
airflow_logger = LoggingMixin().log

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

def load_data(pickled_file_path=PICKLE_FILE_PATH):
    """
    Load data from a pickle file, convert it to a DataFrame, and save as CSV.
    """
    try:
        custom_log("Starting data loading process")
        custom_log(f"Using pickle file: {pickled_file_path}")

        # Load the pickle file
        with open(pickled_file_path, 'rb') as f:
            loaded_data = pickle.load(f)
        custom_log("Pickle file loaded successfully")

        # Convert the loaded data to a DataFrame and store it
        if isinstance(loaded_data, bytes):
            csv_file = io.StringIO(loaded_data.decode('utf-8'))
            df = pd.read_csv(csv_file, sep=';')
            
            # Save as CSV
            df.to_csv(OUTPUT_FILE_PATH, index=False)
            custom_log(f"Data saved as CSV to: {OUTPUT_FILE_PATH}")
            
            return OUTPUT_FILE_PATH
        else:
            custom_log("Loaded data is not in bytes format as expected", level=logging.ERROR)
            return False

    except FileNotFoundError:
        custom_log(f"Pickle file not found: {pickled_file_path}", level=logging.ERROR)
        return False
    except pd.errors.EmptyDataError:
        custom_log("The CSV data in the pickle file is empty", level=logging.ERROR)
        return False
    except Exception as e:
        custom_log(f"An unexpected error occurred: {e}", level=logging.ERROR)
        return False

# This part will only run when the script is executed directly (not through Airflow)
if __name__ == "__main__":
    result = load_data()
    if result:
        custom_log(f"Data loading process completed successfully. Output file: {result}")
    else:
        custom_log("Data loading process failed", level=logging.ERROR)