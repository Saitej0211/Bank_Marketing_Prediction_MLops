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
from src.DownloadData import download_data_from_gcp
from src.LoadData import load_data

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


# ADD NEW TASKS HERE
download_task >> load_task


if __name__ == "__main__":
    dag.cli()