import os
import pandas as pd
import logging

# Set up logging to file
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(PROJECT_DIR, "..", "logs")
LOG_FILE_PATH = os.path.join(LOG_DIR, "process_data.log")

# Ensure the log directory exists
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),  # Log to file
        logging.StreamHandler()  # Also log to console
    ]
)

# Define paths
DATA_DIR = os.path.join(PROJECT_DIR, "..", "data", "processed")
INPUT_FILE_PATH = os.path.join(DATA_DIR, "raw_data.csv")
INFO_CSV_PATH_BEFORE = os.path.join(DATA_DIR, "datatype_info_before.csv")
INFO_CSV_PATH_AFTER = os.path.join(DATA_DIR, "datatype_info_after.csv")
OUTPUT_FILE_PATH = os.path.join(DATA_DIR, "datatype_format_processed.csv")  # New output file path

# Function to check and handle data types
def handle_data_types(data):
    logging.info("Checking data types of the dataset.")
    
    # Log and save the data types before conversion
    data_types_before = data.dtypes
    logging.info(f"Data types BEFORE conversion:\n{data_types_before}")
    data_types_before.to_csv(INFO_CSV_PATH_BEFORE, header=True)
    logging.info(f"Saved BEFORE conversion data type information to {INFO_CSV_PATH_BEFORE}")
    
    for column in data.columns:
        if pd.api.types.is_numeric_dtype(data[column]):
            data[column] = pd.to_numeric(data[column], errors='coerce')
            logging.info(f"Ensured {column} is numeric.")
            
        elif pd.api.types.is_string_dtype(data[column]) or data[column].dtype == 'object':
            data[column] = data[column].astype('string').str.strip().str.lower()  # Explicitly set to "string"
            logging.info(f"Formatted {column} as a string with lowercase and stripped whitespace.")
            
        elif pd.api.types.is_datetime64_any_dtype(data[column]):
            data[column] = pd.to_datetime(data[column], errors='coerce')
            logging.info(f"Converted {column} to datetime.")
        else:
            logging.warning(f"Unhandled data type for column: {column}")
    
    # Log and save the data types after conversion
    data_types_after = data.dtypes
    logging.info(f"Data types AFTER conversion:\n{data_types_after}")
    data_types_after.to_csv(INFO_CSV_PATH_AFTER, header=True)
    logging.info(f"Saved AFTER conversion data type information to {INFO_CSV_PATH_AFTER}")
    
    return data

# Example usage of the function
if __name__ == "__main__":
    try:
        # Load the CSV data
        if os.path.exists(INPUT_FILE_PATH):
            data = pd.read_csv(INPUT_FILE_PATH)
            logging.info(f"Loaded data from {INPUT_FILE_PATH} with shape {data.shape}")
        else:
            logging.error(f"File {INPUT_FILE_PATH} not found.")
            raise FileNotFoundError(f"{INPUT_FILE_PATH} not found.")
        
        # Check and handle data types
        data = handle_data_types(data)

        # Save the processed data to a new CSV file
        data.to_csv(OUTPUT_FILE_PATH, index=False)  # Save the processed data
        logging.info(f"Saved processed data to {OUTPUT_FILE_PATH}")

    except Exception as e:
        logging.error(f"Error occurred: {e}")