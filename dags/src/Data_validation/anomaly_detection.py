import os
import pandas as pd
import json
import logging

# Define paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = os.path.join(PROJECT_DIR, "data", "processed")
LOG_DIR = os.path.join(PROJECT_DIR, "dags", "logs")

INPUT_FILE_PATH = os.path.join(DATA_DIR, "validate.pkl")
OUTPUT_PICKLE_PATH = os.path.join(DATA_DIR, "validate_process.pkl")
OUTPUT_CSV_PATH = os.path.join(DATA_DIR, "validate_process.csv")
OUTPUT_JSON_PATH = os.path.join(DATA_DIR, "validate_process.json")
LOG_FILE_PATH = os.path.join(LOG_DIR, "anomaly_detection.log")

# Ensure necessary directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Set up logger
logger = logging.getLogger('data_processing')
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(LOG_FILE_PATH, mode='a')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def read_data(input_file_path):
    """Read a CSV or pickle file and return it as a DataFrame."""
    try:
        logger.info("Starting data loading process")
        if os.path.exists(input_file_path):
            if input_file_path.endswith('.csv'):
                df = pd.read_csv(input_file_path, encoding='ISO-8859-1')
                logger.info(f"Loaded CSV data with shape {df.shape}")
            elif input_file_path.endswith('.pkl'):
                df = pd.read_pickle(input_file_path)
                logger.info(f"Loaded pickle data with shape {df.shape}")
            else:
                logger.error(f"Unsupported file type: {input_file_path}")
                raise ValueError(f"Unsupported file type: {input_file_path}")
            return df
        else:
            logger.error(f"File {input_file_path} not found.")
            raise FileNotFoundError(f"{input_file_path} not found.")
    except Exception as e:
        logger.exception(f"Error in read_data: {e}")
        raise

def save_dataframe(df, output_pickle_path, output_csv_path):
    """Save a DataFrame as both a pickle and CSV file."""
    try:
        df.to_pickle(output_pickle_path)
        logger.info(f"Data successfully saved to {output_pickle_path}")
        df.to_csv(output_csv_path, index=False)
        logger.info(f"Data successfully saved to {output_csv_path}")
    except Exception as e:
        logger.exception(f"Failed to save DataFrame: {e}")
        raise

def anomalyDetect(input_path):
    # Load data
    data = read_data(input_path)
    
    # Initialize results dictionary
    results = {
        "stats": {},
        "issues": {}
    }

    # Calculate statistics
    results['stats']['num_rows'] = data.shape[0]
    results['stats']['num_columns'] = data.shape[1]
    results['stats']['numeric_summary'] = data.describe().to_dict()
    
    # Expected norms and thresholds
    min_rows = 10
    expected_columns = {
        'age': 'int64', 'job': 'object', 'marital': 'object', 'education': 'object', 
        'default': 'object', 'balance': 'int64', 'housing': 'object', 'loan': 'object', 
        'contact': 'object', 'day': 'int64', 'month': 'object', 'duration': 'int64', 
        'campaign': 'int64', 'pdays': 'int64', 'previous': 'int64', 'poutcome': 'object', 'y': 'object'
    }
    allowed_values = {
        'default': ['yes', 'no'], 'housing': ['yes', 'no'], 'loan': ['yes', 'no'], 
        'y': ['yes', 'no'], 'marital': ['married', 'single', 'divorced'],
        'education': ['primary', 'secondary', 'tertiary', 'unknown'],
        'contact': ['unknown', 'telephone', 'cellular'],
        'poutcome': ['unknown', 'other', 'failure', 'success'],
        'month': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    }
    
    # Check row count
    if results['stats']['num_rows'] < min_rows:
        results['issues']['row_count_issue'] = f"Expected at least {min_rows} rows, found {results['stats']['num_rows']}"
        logger.error(results['issues']['row_count_issue'])
    
    # Data type and value checks
    for column, expected_dtype in expected_columns.items():
        if column in data.columns:
            # Check data type
            if data[column].dtype != expected_dtype:
                results['issues'][f'{column}_dtype_issue'] = f"Expected {expected_dtype}, found {data[column].dtype}"
                logger.error(results['issues'][f'{column}_dtype_issue'])
            # Check categorical values
            if column in allowed_values:
                invalid_values = set(data[column].unique()) - set(allowed_values[column])
                if invalid_values:
                    results['issues'][f'{column}_invalid_values'] = f"Unexpected values found: {invalid_values}"
                    logger.error(results['issues'][f'{column}_invalid_values'])
        else:
            results['issues'][f'{column}_missing'] = "Column is missing in the dataset"
            logger.error(results['issues'][f'{column}_missing'])
    
    # Numeric range checks
    if (data['age'] < 0).any():
        results['issues']['age_negative'] = "Found negative values in 'age'"
        logger.error(results['issues']['age_negative'])

    # Save the results as pickle and CSV
    save_dataframe(data, OUTPUT_PICKLE_PATH, OUTPUT_CSV_PATH)
        
    return results

def anomaly_detection():
    """Main function to handle data loading, processing, and saving."""
    try:
        # Detect anomalies in data
        anomaly_results = anomalyDetect(INPUT_FILE_PATH)

        # Save the results as JSON
        with open(OUTPUT_JSON_PATH, 'w') as json_file:
            json.dump(anomaly_results, json_file, indent=4)
        logger.info(f"Data successfully saved to {OUTPUT_JSON_PATH}")
        
        # Log the JSON output of anomaly results
        logger.info("Anomaly detection completed.")
        logger.info(json.dumps(anomaly_results, indent=4))
        
        print(json.dumps(anomaly_results, indent=4))

        # Return anomaly_results for XCom
        return anomaly_results  # This ensures the results are available in XCom
        
    except Exception as e:
        logger.error(f"An error occurred during data processing: {e}")

if __name__ == "__main__":
    anomaly_detection()
