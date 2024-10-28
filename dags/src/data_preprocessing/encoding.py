import io
import os
import pandas as pd
import logging
from sklearn.preprocessing import LabelEncoder

from airflow.utils.log.logging_mixin import LoggingMixin


# Define paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = os.path.join(PROJECT_DIR, "data", "processed")


#INPUT_FILE_PATH = os.path.join(DATA_DIR, "outlier_handled_data.pkl") # Data after the outliers are handled
OUTPUT_PICKLE_PATH = os.path.join(DATA_DIR, "encoded_data.pkl")  # New output file path after encoding
OUTPUT_CSV_PATH  = os.path.join(DATA_DIR, "encoded_data.csv")  # encoded csv output path

# Set up Airflow logger
airflow_logger = LoggingMixin().log

'''
# Set up logging
LOG_FILE_PATH = os.path.join(LOG_DIR, "encoding.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler()
    ]
)
'''

LOG_DIR = os.path.join(PROJECT_DIR, "dags", "logs")

# Ensure necessary directories exist
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOG_DIR, 'encoding.log')

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


def encode_categorical_variables(input_file_path):
    try:
        custom_log("Starting categorical encoding process")

        # Load data based on file extension
        if os.path.exists(input_file_path):
            if input_file_path.endswith('.csv'):
                df = pd.read_csv(input_file_path)
            elif input_file_path.endswith('.pkl'):
                # Attempt to read as pickle and catch potential errors
                try:
                    df = pd.read_pickle(input_file_path)
                    custom_log(f"Loaded data from {input_file_path} with shape {df.shape}")
                    column_names = df.columns[0].split(',')
                    custom_log(column_names)
            
                    # Split the data into rows and columns
                    df = df.iloc[:, 0].str.split(',', expand=True)
 
                except Exception as e:
                    logging.error(f"Failed to read pickle file: {e}")
                    raise
            else:
                logging.error(f"Unsupported file type: {input_file_path}")
                raise ValueError(f"Unsupported file type: {input_file_path}")

            logging.info(f"Loaded data from {input_file_path} with shape {df.shape}")
        else:
            logging.error(f"File {input_file_path} not found.")
            raise FileNotFoundError(f"{input_file_path} not found.")

        # Assign dynamic column names
        df.columns = column_names

        # Log column datatypes
        custom_log("Column datatypes:")
        for column, dtype in df.dtypes.items():
            custom_log(f"{column}: {dtype}")

        # Identify categorical columns
        categorical_columns = df.select_dtypes(include=['object', 'string', 'category']).columns
        custom_log(f"Identified {len(categorical_columns)} categorical columns: {list(categorical_columns)}")

        # Initialize LabelEncoder
        le = LabelEncoder()

        # Encode categorical variables
        for col in categorical_columns:
            df[col] = le.fit_transform(df[col].astype(str))
            custom_log(f"Encoded column: {col}")

        # Save the encoded data as CSV
        df.to_csv(OUTPUT_CSV_PATH, index=False)
        custom_log(f"Saved encoded data to CSV file: {OUTPUT_CSV_PATH}")

        # Save the encoded data as pickle
        df.to_pickle(OUTPUT_PICKLE_PATH)
        custom_log(f"Saved encoded data to pickle file: {OUTPUT_PICKLE_PATH}")

        custom_log("Categorical encoding process completed successfully")
        return OUTPUT_PICKLE_PATH

    except Exception as e:
        custom_log,(f"An error occurred during categorical encoding: {e}")
        raise

if __name__ == "__main__":
    encode_categorical_variables()