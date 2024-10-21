""""
        AIRFLOW DAG FILE
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
from src.LoadData import load_data_from_gcp

#Define the paths to project directory and the path to the key
PROJECT_DIR = os.getcwd()
data_dir = PROJECT_DIR
bucket_name = "mlopsprojectdatabucketgrp6"

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


#initialize the dag

dag = DAG(
    'DataPipeline',
    default_args = default_args,
    description = 'MLOps Data pipeline',
    schedule_interval = None,  # Set the schedule interval or use None for manual triggering
    catchup = False,
)

#define the tasks that depend on the dunctions created in the src folder
download_and_pickle_task = PythonOperator(
    task_id='download_and_pickle_latest_file',
    python_callable= load_data_from_gcp,
    op_kwargs={
        'data_dir':data_dir,
        'bucket_name': bucket_name
    },
    dag=dag,
)