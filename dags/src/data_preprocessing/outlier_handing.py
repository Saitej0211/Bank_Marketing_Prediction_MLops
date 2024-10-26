import os
import pandas as pd
import logging
import numpy as np
import matplotlib.pyplot as plt

# Define paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = os.path.join(PROJECT_DIR, "data", "processed")
LOG_DIR = os.path.join(PROJECT_DIR, "logs")
OUTPUT_FILE_PATH = os.path.join(DATA_DIR, "outlier_handled_data.pkl")  # New output file path
LOG_FILE_PATH = os.path.join(LOG_DIR, "outlier_handling.log")

# Ensure necessary directories exist
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
# Function to detect and handle outliers using IQR
def handle_outliers(data, threshold=1.5):
    logging.info("Starting outlier handling.")
    
    data_handled = data.copy()  # Make a copy to avoid modifying the original data
    
    for column in data.select_dtypes(include=[np.number]).columns:  # Only numeric columns
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        # Log the outliers
        outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
        logging.info(f"Outliers in {column}:\n{outliers}")
        
        # Cap or replace outliers
        data_handled[column] = np.where(data[column] < lower_bound, lower_bound, data[column])
        data_handled[column] = np.where(data[column] > upper_bound, upper_bound, data[column])
        logging.info(f"Handled outliers in {column} using IQR method.")
        
    logging.info("Outlier handling completed.")
    return data_handled

# Main processing function
def process_outlier_handling(input_file_path):
    """
    Process the input data to handle outliers and save results.
    """
    try:
        logging.info("Starting outlier handling process")
        logging.info(f"Input file: {input_file_path}")

        # Load the processed data (assuming it is a pickle file)
        if os.path.exists(input_file_path):
            data = pd.read_pickle(input_file_path)  # Read as a pickle file
            logging.info(f"Loaded data from {input_file_path} with shape {data.shape}")
        else:
            logging.error(f"File {input_file_path} not found.")
            raise FileNotFoundError(f"{input_file_path} not found.")

        # Handle outliers using IQR method
        data_handled = handle_outliers(data, threshold=1.5)

        # Save the outlier-handled data to a new pickle file
        data_handled.to_pickle(OUTPUT_FILE_PATH)
        logging.info(f"Saved outlier-handled data to {OUTPUT_FILE_PATH}")

        logging.info("Outlier handling process completed successfully")
        return OUTPUT_FILE_PATH  # Return the output file path for further processing

    except Exception as e:
        logging.error(f"Error occurred: {e}")
        raise  # Reraise the exception after logging it

if __name__ == "__main__":
    output_path = process_outlier_handling(os.path.join(DATA_DIR, "datatype_format_processed.pkl"))
    logging.info(f"Output file path: {output_path}")