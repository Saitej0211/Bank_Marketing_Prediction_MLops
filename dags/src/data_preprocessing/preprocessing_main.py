import pandas as pd
import os
from .datatype_format import process_datatype
from .outlier_handing import process_outlier_handling

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = os.path.join(PROJECT_DIR, "data", "processed")
INPUT_FILE_PATH = os.path.join(DATA_DIR, "processed_data.pkl")

def preprocess_data(input_file_path= INPUT_FILE_PATH):
    """
    Main preprocessing method that handles outliers and formats data types.

    :param input_file_path: Path to the input data file.
    :param output_file_path: Path to save the processed data file.
    """

    # Format data types
    data = process_datatype(input_file_path)

    # Outlier handling
    data = process_outlier_handling(data)

    
    print(f"Processed data saved to {data}")
    return data

if __name__ == "__main__":
    preprocess_data(INPUT_FILE_PATH)