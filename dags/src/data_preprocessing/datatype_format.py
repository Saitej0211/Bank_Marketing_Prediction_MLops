import os
import pandas as pd
import logging

# Define paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = os.path.join(PROJECT_DIR, "data", "processed")
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Input and output file paths
INPUT_FILE_PATH = os.path.join(DATA_DIR, "processed_data.pkl")  # Input from HandlingNullValues
INFO_CSV_PATH_BEFORE = os.path.join(DATA_DIR, "datatype_info_before.csv")
INFO_CSV_PATH_AFTER = os.path.join(DATA_DIR, "datatype_info_after.csv")
OUTPUT_PICKLE_PATH = os.path.join(DATA_DIR, "datatype_format_processed.pkl")  # New output pickle path
LOG_FILE_PATH = os.path.join(LOG_DIR,"logs", "process_data.log")

# Ensure necessary directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),  # Log to file
        logging.StreamHandler()  # Also log to console
    ]
)

# Function to check and handle data types
def handle_data_types(data):
    logging.info("Checking data types of the dataset.")

    # Log and save the data types before conversion
    data_types_before = data.dtypes
    data_types_before.to_csv(INFO_CSV_PATH_BEFORE, header=True)
    logging.info(f"Saved BEFORE conversion data type information to {INFO_CSV_PATH_BEFORE}")

    for column in data.columns:
        if pd.api.types.is_numeric_dtype(data[column]):
            data[column] = pd.to_numeric(data[column], errors='coerce')
        elif data[column].dtype == 'object':
            data[column] = data[column].astype('string').str.strip().str.lower()  # Format strings
        elif pd.api.types.is_datetime64_any_dtype(data[column]):
            data[column] = pd.to_datetime(data[column], errors='coerce')
        else:
            logging.warning(f"Unhandled data type for column: {column}")

    # Log and save the data types after conversion
    data_types_after = data.dtypes
    data_types_after.to_csv(INFO_CSV_PATH_AFTER, header=True)
    logging.info(f"Saved AFTER conversion data type information to {INFO_CSV_PATH_AFTER}")

    return data

# Main processing function
# Main processing function
# Main processing function
def process_datatype(input_file_path=INPUT_FILE_PATH):
    """
    Process the input CSV data, perform data type handling, and save results.
    """
    try:
        logging.info("Starting data processing")
        logging.info(f"Input file: {input_file_path}")

        # Load data based on file extension
        if os.path.exists(input_file_path):
            if input_file_path.endswith('.csv'):
                data = pd.read_csv(input_file_path)
            elif input_file_path.endswith('.pkl'):
                # Attempt to read as pickle and catch potential errors
                try:
                    data = pd.read_pickle(input_file_path)
                except Exception as e:
                    logging.error(f"Failed to read pickle file: {e}")
                    raise
            else:
                logging.error(f"Unsupported file type: {input_file_path}")
                raise ValueError(f"Unsupported file type: {input_file_path}")

            logging.info(f"Loaded data from {input_file_path} with shape {data.shape}")
        else:
            logging.error(f"File {input_file_path} not found.")
            raise FileNotFoundError(f"{input_file_path} not found.")

        # Check and handle data types
        data = handle_data_types(data)

        # Save the processed data as a pickle file
        data.to_pickle(OUTPUT_PICKLE_PATH)
        logging.info(f"Processed data saved as pickle at {OUTPUT_PICKLE_PATH}")

        logging.info("Data processing completed successfully")
        return OUTPUT_PICKLE_PATH  # Return the output file path for further processing

    except Exception as e:
        logging.error(f"Error occurred: {e}")
        raise  # Reraise the exception after logging it

if __name__ == "__main__":
    output_path = process_datatype()
    logging.info(f"Output file path: {output_path}")