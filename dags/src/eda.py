import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging
from airflow.utils.log.logging_mixin import LoggingMixin

# Define paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_DIR, "data", "processed")
OUTPUT_DIR = os.path.join(DATA_DIR, "eda_plots")
LOG_DIR = os.path.join(PROJECT_DIR,"dags","logs")
LOG_FILE_PATH = os.path.join(LOG_DIR, "eda.log")

# Ensure necessary directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
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

def save_plot(fig, filename):
    """Save the current plot to a file and close it."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath)
    plt.close(fig)
    custom_log(f"Saved plot: {filepath}")

def perform_eda(input_file_path):
    """Perform EDA tasks and save plots."""
    try:
        custom_log("Starting EDA process")
        custom_log(f"Using input file: {input_file_path}")

        # Load the data
        data = pd.read_pickle(input_file_path)
        custom_log(f"Loaded data from {input_file_path} with shape {data.shape}")
        custom_log(f"Data head:\n{data.head().to_string()}")
        
        # Print current column names
        custom_log(f"Current column names: {data.columns.tolist()}")

        # Target values pie chart
        plt.figure(figsize=(8,8))
        data['y'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90)
        plt.title('Deposit Distribution')
        save_plot(plt.gcf(), "deposit_distribution.png")
        
        # Contact method distribution
        contact_dist = data['contact'].value_counts(normalize=True)
        custom_log(f"Contact method distribution:\n{contact_dist}")
        
        # Contact method countplot
        plt.figure(figsize=(10,6))
        data['contact'].value_counts().plot(kind='bar')
        plt.title('Contact Method Distribution')
        plt.xlabel('Contact Method')
        plt.ylabel('Count')
        save_plot(plt.gcf(), "contact_method_distribution.png")
        
        # Housing loan countplot
        plt.figure(figsize=(8,6))
        data.groupby(['housing', 'y']).size().unstack().plot(kind='bar', stacked=True)
        plt.title('Housing Loan Distribution by Deposit')
        plt.xlabel('Housing Loan')
        plt.ylabel('Count')
        plt.legend(title='Deposit', labels=['No', 'Yes'])
        save_plot(plt.gcf(), "housing_loan_distribution.png")
        
        # Personal loan countplot
        plt.figure(figsize=(8,6))
        data.groupby(['loan', 'y']).size().unstack().plot(kind='bar', stacked=True)
        plt.title('Personal Loan Distribution by Deposit')
        plt.xlabel('Personal Loan')
        plt.ylabel('Count')
        plt.legend(title='Deposit', labels=['No', 'Yes'])
        save_plot(plt.gcf(), "personal_loan_distribution.png")
        
        # Default countplot
        plt.figure(figsize=(8,6))
        data.groupby(['default', 'y']).size().unstack().plot(kind='bar', stacked=True)
        plt.title('Default Status Distribution by Deposit')
        plt.xlabel('Default Status')
        plt.ylabel('Count')
        plt.legend(title='Deposit', labels=['No', 'Yes'])
        save_plot(plt.gcf(), "default_status_distribution.png")
        
        # Month distribution
        month_dist = data['month'].value_counts(normalize=True)
        custom_log(f"Month distribution:\n{month_dist}")
        
        # Month countplot
        plt.figure(figsize=(12,6))
        data.groupby(['month', 'y']).size().unstack().plot(kind='bar', stacked=True)
        plt.title('Month Distribution by Deposit')
        plt.xlabel('Month')
        plt.ylabel('Count')
        plt.legend(title='Deposit', labels=['No', 'Yes'])
        plt.xticks(rotation=45)
        save_plot(plt.gcf(), "month_distribution.png")
        
        # Age distribution
        plt.figure(figsize=(10,6))
        plt.hist(data['age'], bins=30, edgecolor='black')
        plt.title('Age Distribution')
        plt.xlabel('Age')
        plt.ylabel('Count')
        save_plot(plt.gcf(), "age_distribution.png")
        
        # Job distribution
        plt.figure(figsize=(12, 6))
        data['job'].value_counts().plot(kind='bar')
        plt.title('Job Distribution')
        plt.xlabel('Job')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        save_plot(plt.gcf(), "job_distribution.png")
        
        # Marital status countplot
        plt.figure(figsize=(8,6))
        data.groupby(['marital', 'y']).size().unstack().plot(kind='bar', stacked=True)
        plt.title('Distribution of Marital Status by Deposit')
        plt.xlabel('Marital Status')
        plt.ylabel('Count')
        plt.legend(title='Deposit', labels=['No', 'Yes'])
        save_plot(plt.gcf(), "marital_status_distribution.png")
        
        # Education countplot
        plt.figure(figsize=(10,6))
        data.groupby(['education', 'y']).size().unstack().plot(kind='bar', stacked=True)
        plt.title('Distribution of Education among Customers')
        plt.xlabel('Education')
        plt.ylabel('Count')
        plt.legend(title='Deposit', labels=['No', 'Yes'])
        plt.xticks(rotation=45)
        save_plot(plt.gcf(), "education_distribution.png")
        
        # Correlation heatmap
        plt.figure(figsize=(12,10))
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        corr = data[numeric_columns].corr()
        plt.imshow(corr, cmap='coolwarm')
        plt.colorbar()
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.columns)), corr.columns)
        plt.title('Correlation Heatmap (Numeric Columns Only)')
        
        # Add correlation values to the heatmap
        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                plt.text(j, i, f"{corr.iloc[i, j]:.2f}", 
                         ha="center", va="center", color="black")
        
        plt.tight_layout()
        save_plot(plt.gcf(), "correlation_heatmap.png")
        
        custom_log("EDA process completed successfully")
    
    except Exception as e:
        custom_log(f"Error occurred during EDA: {e}", level=logging.ERROR)
        raise

if __name__ == "__main__":
    INPUT_FILE_PATH = os.path.join(DATA_DIR, "outlier_handled_data.pkl")
    perform_eda(INPUT_FILE_PATH)