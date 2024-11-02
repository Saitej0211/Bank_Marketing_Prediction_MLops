""""
        AIRFLOW DAG FILE FOR DATA PREPROCESSING
"""

#import all libraries
from datetime import datetime, timedelta
from airflow import DAG
from airflow import configuration as conf
from airflow.operators.python import PythonOperator
from airflow.operators.email import EmailOperator
from airflow.operators.bash import BashOperator
import os

#import all the functions that we had created in the src folder
from src.data_preprocessing.smote import smote_analysis
from src.data_preprocessing.correlation_analysis import correlation_analysis
from src.data_preprocessing.encoding import encode_categorical_variables
from src.DownloadData import download_data_from_gcp
from src.LoadData import load_data
from src.HandlingNullValues import process_data
from src.data_preprocessing.preprocessing_main import preprocess_data
from src.eda import perform_eda
from src.Data_validation.data_schema_statistics_generation import validate_data_schema
from src.Data_validation.anomaly_detection import anomaly_detection

#Define the paths to project directory and the path to the key
PROJECT_DIR = os.getcwd()

#BUCKET NAME
BUCKET_NAME = "mlopsprojectdatabucketgrp6"

# Enable xcom pickling to allow passage of tasks from one task to another.
conf.set('core', 'enable_xcom_pickling', 'True')

# Set default arguments

default_args = {
    'owner': 'MLopsProjectGroup6',
    'depends_on_past': False,
    'start_date': datetime(2024,10,19),
    'retries': 2,
    'retry_delay':timedelta(minutes=5)
}

# Email settings
EMAIL_RECIPIENT = "hansda.s@northeastern.edu"


#INITIALIZE THE DAG INSTANCE
dag = DAG(
    'DataPipeline',
    default_args = default_args,
    description = 'MLOps Data pipeline',
    schedule_interval = None,  # Set the schedule interval or use None for manual triggering
    catchup = False,
)

#DEFINE TASKS THAT FORM THE COMPONENTS OF THE DAG

#DEFINE A FUNCTION TO DOWNLOAD THE DATA FROM GCP
download_task = PythonOperator(
    task_id='download_data_from_gcp',
    python_callable=download_data_from_gcp,
    op_kwargs={'bucket_name': 'mlopsprojectdatabucketgrp6'},
    dag=dag,
)

load_task = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    op_kwargs={
        'pickled_file_path': '{{ ti.xcom_pull(task_ids="download_data_from_gcp") }}',
    },
    dag=dag,
)

# Task to process data and drop columns with >80% null values
process_task = PythonOperator(
    task_id='process_data',
    python_callable=process_data,
    op_kwargs={
        'pickled_file_path': '{{ ti.xcom_pull(task_ids="load_task") }}',
    },
    dag=dag,
)

# Task to process data with data type formatting and outlier handling
pre_process_task = PythonOperator(
    task_id='pre_process_data',
    python_callable=preprocess_data,
    op_kwargs={
        'pickled_file_path': '{{ ti.xcom_pull(task_ids="process_data") }}',
    },
    dag=dag,
)

# Task to perform EDA
eda_task = PythonOperator(
    task_id='perform_eda',
    python_callable=perform_eda,
    op_kwargs={
        'input_file_path': '{{ ti.xcom_pull(task_ids="pre_process_data") }}',
    },
    dag=dag,
)

# Task to perform encoding of categorical variables
encode_categorical_task = PythonOperator(
    task_id='encode_categorical_variables',
    python_callable=encode_categorical_variables,
    op_kwargs={
        'input_file_path': '{{ ti.xcom_pull(task_ids="pre_process_data") }}',
    },
    dag=dag,
)

correlation_analysis_task = PythonOperator(
    task_id='correlation_analysis',
    python_callable=correlation_analysis,
    op_kwargs={'input_file_path': '{{ ti.xcom_pull(task_ids="encode_categorical_variables") }}'},
    dag=dag,
)

smote_analysis_task = PythonOperator(
    task_id='smote_analysis',
    python_callable=smote_analysis,
    op_kwargs={'input_file_path': '{{ ti.xcom_pull(task_ids="encode_categorical_variables") }}'},
    dag=dag,
)

validate_task_1 = PythonOperator(
    task_id='validate_data_schema',
    python_callable=validate_data_schema,
    op_kwargs={
        'pickled_file_path': '{{ ti.xcom_pull(task_ids="load_data") }}',
    },
    dag=dag,
)
 
validate_task_2 = PythonOperator(
    task_id='anomaly_detection',
    python_callable = anomaly_detection,
    op_kwargs={
        'pickled_file_path': '{{ ti.xcom_pull(task_ids="validate_data_schema") }}',
    },
    dag=dag,
)

# Define task dependencies
download_task >> load_task >> validate_task_1 >> validate_task_2 >> process_task >> pre_process_task >> eda_task >> encode_categorical_task >> correlation_analysis_task >> smote_analysis_task


if __name__ == "__main__":
    dag.cli()