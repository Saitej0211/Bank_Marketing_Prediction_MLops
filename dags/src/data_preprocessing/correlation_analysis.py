import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from airflow.utils.log.logging_mixin import LoggingMixin

# Define paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = os.path.join(PROJECT_DIR, "data", "processed")
EDA_PLOTS_DIR = os.path.join(DATA_DIR, "eda_plots")
LOG_DIR = os.path.join(PROJECT_DIR, "dags", "logs")

# Ensure necessary directories exist
os.makedirs(EDA_PLOTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Set up Airflow logger
airflow_logger = LoggingMixin().log

# Set up file logger
import logging
LOG_FILE_PATH = os.path.join(LOG_DIR, 'correlation_analysis.log')
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

def correlation_analysis(input_file_path):
    try:
        custom_log("Starting correlation analysis")
        custom_log(f"Input file: {input_file_path}")

        # Load the encoded data
        df = pd.read_pickle(input_file_path)
        custom_log(f"Loaded data from {input_file_path} with shape {df.shape}")

        # Compute correlation matrix
        correlation_matrix = df.corr()

        # Save correlation matrix as CSV
        correlation_csv_path = os.path.join(DATA_DIR, "correlation_matrix.csv")
        correlation_matrix.to_csv(correlation_csv_path)
        custom_log(f"Correlation matrix saved to {correlation_csv_path}")

        # Save correlation matrix as pickle
        correlation_pkl_path = os.path.join(DATA_DIR, "correlation_matrix.pkl")
        correlation_matrix.to_pickle(correlation_pkl_path)
        custom_log(f"Correlation matrix saved to {correlation_pkl_path}")

        # Log highly correlated pairs (absolute correlation > 0.8)
        high_corr = correlation_matrix[(correlation_matrix.abs() > 0.8) & (correlation_matrix != 1.0)]
        high_corr_pairs = [(col1, col2, corr_value) for col1, row in high_corr.iterrows() for col2, corr_value in row.items() if pd.notna(corr_value)]

        if high_corr_pairs:
            custom_log("Highly correlated pairs (correlation > 0.8):")
            for col1, col2, corr_value in high_corr_pairs:
                custom_log(f"{col1} - {col2}: {corr_value}")
        else:
            custom_log("No highly correlated pairs found.")

        # Create correlation heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
        plt.title('Correlation Heatmap')
        
        # Save correlation plot
        correlation_plot_path = os.path.join(EDA_PLOTS_DIR, "correlation_heatmap.png")
        plt.savefig(correlation_plot_path)
        plt.close()
        custom_log(f"Correlation heatmap saved to {correlation_plot_path}")

        custom_log("Correlation analysis completed successfully")
        return correlation_pkl_path

    except Exception as e:
        custom_log(f"An error occurred during correlation analysis: {e}", level=logging.ERROR)
        raise

if __name__ == "__main__":
    #input_file_path = os.path.join(DATA_DIR, "encoded_data.pkl")  # Adjust this path if needed
    correlation_analysis()