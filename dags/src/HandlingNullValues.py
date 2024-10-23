import os
import logging
import pandas as pd
import io

# Define paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Adjust if needed
DATA_DIR = os.path.join(PROJECT_DIR, "..", "data", "processed")
INPUT_FILE_PATH = os.path.join(DATA_DIR, "raw_data.csv")
PICKLE_FILE_PATH = os.path.join(DATA_DIR, "processed_data.pkl")
INFO_CSV_PATH = os.path.join(DATA_DIR, "dataframe_info.csv")
DESCRIPTION_CSV_PATH = os.path.join(DATA_DIR, "dataframe_description.csv")

def process_data(input_file_path=INPUT_FILE_PATH):
    try:
        # Set up logging
        logs_dir = os.path.join(PROJECT_DIR, "..", "logs")
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        
        log_file_path = os.path.join(logs_dir, 'process_data.log')
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            handlers=[logging.FileHandler(log_file_path, mode='w'),
                                      logging.StreamHandler()])
        
        logging.info("Loading CSV data for processing")

        # Load CSV data directly
        df = pd.read_csv(input_file_path, sep=';')

        # Log and save DataFrame shape
        logging.info(f"DataFrame shape: {df.shape}")

        # Capture and save DataFrame info
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue().strip().split('\n')
        info_df = pd.DataFrame(info_str[2:], columns=['Info'])
        info_df.to_csv(INFO_CSV_PATH, index=False)
        logging.info(f"DataFrame info saved to {INFO_CSV_PATH}")

        # Capture and save DataFrame description
        description_df = df.describe()
        description_df.to_csv(DESCRIPTION_CSV_PATH)
        logging.info(f"DataFrame description saved to {DESCRIPTION_CSV_PATH}")

        # Check for duplicate rows
        duplicate_rows = df.duplicated().sum()
        if duplicate_rows > 0:
            logging.info(f"Found {duplicate_rows} duplicate rows. Dropping duplicates.")
            df = df.drop_duplicates()
            logging.info("Duplicate rows dropped.")
        else:
            logging.info("No duplicate rows found.")

        # Calculate percentage of null values
        null_percentage = df.isnull().mean() * 100

        # Identify features to drop
        features_to_drop = null_percentage[null_percentage > 50].index.tolist()

        # Drop the features
        df_dropped = df.drop(columns=features_to_drop)

        # Log dropped features
        log_message = f'Dropped features: {features_to_drop}' if features_to_drop else 'No features dropped'
        logging.info(log_message)

        # Save processed data as a pickle file
        df_dropped.to_pickle(PICKLE_FILE_PATH)
        logging.info(f"Processed data saved as pickle at {PICKLE_FILE_PATH}")

    except Exception as e:
        logging.error(f"An unexpected error occurred during data processing: {e}")

if __name__ == "__main__":
    process_data(INPUT_FILE_PATH)