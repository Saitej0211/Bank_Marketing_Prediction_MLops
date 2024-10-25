import os
import pandas as pd
import logging
import numpy as np
import matplotlib.pyplot as plt

# Set up logging to file
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(PROJECT_DIR, "..", "logs")
LOG_FILE_PATH = os.path.join(LOG_DIR, "outlier_handling.log")

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
INPUT_FILE_PATH = os.path.join(DATA_DIR, "datatype_format_processed.csv")
OUTPUT_FILE_PATH = os.path.join(DATA_DIR, "outlier_handled_data.csv")  # New output file path
PLOTS_DIR = os.path.join(PROJECT_DIR, "..", "assets", "plots")  # Directory for saving plots
os.makedirs(PLOTS_DIR, exist_ok=True)

# Function to plot data before and after outlier handling
def plot_distribution(data_before, data_after, column):
    plt.figure(figsize=(10, 6))
    
    # Plot before handling outliers
    plt.subplot(1, 2, 1)
    plt.hist(data_before[column], bins=30, color='blue', alpha=0.7, label='Before')
    plt.title(f'{column} - Before Outlier Handling')
    plt.xlabel(column)
    plt.ylabel('Frequency')

    # Plot after handling outliers
    plt.subplot(1, 2, 2)
    plt.hist(data_after[column], bins=30, color='green', alpha=0.7, label='After')
    plt.title(f'{column} - After Outlier Handling')
    plt.xlabel(column)
    plt.ylabel('Frequency')

    # Save plot
    plot_file_path = os.path.join(PLOTS_DIR, f"{column}_outlier_handling.png")
    plt.tight_layout()
    plt.savefig(plot_file_path)
    plt.close()  # Close the plot to free up memory
    logging.info(f"Saved plot for {column} before and after outlier handling to {plot_file_path}")

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
        
        # Plot the data before and after outlier handling
        plot_distribution(data, data_handled, column)

    logging.info("Outlier handling completed.")
    return data_handled

# Example usage of the function
if __name__ == "__main__":
    try:
        # Load the processed CSV data
        if os.path.exists(INPUT_FILE_PATH):
            data = pd.read_csv(INPUT_FILE_PATH)
            logging.info(f"Loaded data from {INPUT_FILE_PATH} with shape {data.shape}")
        else:
            logging.error(f"File {INPUT_FILE_PATH} not found.")
            raise FileNotFoundError(f"{INPUT_FILE_PATH} not found.")
        
        # Handle outliers using IQR method
        data_handled = handle_outliers(data, threshold=1.5)

        # Save the outlier-handled data to a new CSV file
        data_handled.to_csv(OUTPUT_FILE_PATH, index=False)
        logging.info(f"Saved outlier-handled data to {OUTPUT_FILE_PATH}")

    except Exception as e:
        logging.error(f"Error occurred: {e}")