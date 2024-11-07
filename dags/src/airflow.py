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
from airflow.utils.email import send_email
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

import os
import logging

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

# Define function to notify failure or sucess via an email
def notify_success(context):
    success_email = EmailOperator(
        task_id='success_email',
        to='aishwariya.alagesanus@gmail.com',
        subject='Success Notification from Airflow',
        html_content='<p>The task succeeded.</p>',
        dag=context['dag']
    )
    success_email.execute(context=context)

def notify_failure(context):
    failure_email = EmailOperator(
        task_id='failure_email',
        to='aishwariya.alagesanus@gmail.com',
        subject='Failure Notification from Airflow',
        html_content='<p>The task failed.</p>',
        dag=context['dag']
    )
    failure_email.execute(context=context)

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
    on_failure_callback=notify_failure,
    dag=dag,
)

load_task = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    op_kwargs={
        'pickled_file_path': '{{ ti.xcom_pull(task_ids="download_data_from_gcp") }}',
    },
    on_failure_callback=notify_failure,
    dag=dag,
)

# Task to process data and drop columns with >80% null values
pre_process_task1 = PythonOperator(
    task_id='pre_process_task1',
    python_callable=process_data,
    op_kwargs={
        'pickled_file_path': '{{ ti.xcom_pull(task_ids="load_task") }}',
    },
    on_failure_callback=notify_failure,
    dag=dag,
)

# Task to process data with data type formatting and outlier handling
pre_process_task2 = PythonOperator(
    task_id='pre_process_task2',
    python_callable=preprocess_data,
    op_kwargs={
        'pickled_file_path': '{{ ti.xcom_pull(task_ids="pre_process_task1") }}',
    },
    on_failure_callback=notify_failure,
    dag=dag,
)

# Task to perform EDA
eda_task = PythonOperator(
    task_id='perform_eda',
    python_callable=perform_eda,
    op_kwargs={
        'input_file_path': '{{ ti.xcom_pull(task_ids="pre_process_task2") }}',
    },
    on_failure_callback=notify_failure,
    dag=dag,
)

# Task to perform encoding of categorical variables
encode_categorical_task = PythonOperator(
    task_id='encode_categorical_variables',
    python_callable=encode_categorical_variables,
    op_kwargs={
        'input_file_path': '{{ ti.xcom_pull(task_ids="pre_process_task2") }}',
    },
    on_failure_callback=notify_failure,
    dag=dag,
)

correlation_analysis_task = PythonOperator(
    task_id='correlation_analysis',
    python_callable=correlation_analysis,
    op_kwargs={'input_file_path': '{{ ti.xcom_pull(task_ids="encode_categorical_variables") }}'},
    on_failure_callback=notify_failure,
    dag=dag,
)

smote_analysis_task = PythonOperator(
    task_id='smote_analysis',
    python_callable=smote_analysis,
    op_kwargs={'input_file_path': '{{ ti.xcom_pull(task_ids="encode_categorical_variables") }}'},
    on_failure_callback=notify_failure,
    dag=dag,
)

stats_validate_task = PythonOperator(
    task_id='validate_data_schema',
    python_callable=validate_data_schema,
    op_kwargs={
        'pickled_file_path': '{{ ti.xcom_pull(task_ids="load_data") }}',
    },
    on_failure_callback=notify_failure,
    dag=dag,
)
 
anomaly_validate_task = PythonOperator(
    task_id='anomaly_detection',
    python_callable = anomaly_detection,
    op_kwargs={
        'pickled_file_path': '{{ ti.xcom_pull(task_ids="validate_data_schema") }}',
    },
    on_failure_callback=notify_failure,
    dag=dag,
)

send_alert_to = "aishwariya.alagesanus@gmail.com"

def check_anomalies_and_alert(**kwargs):
    ti = kwargs['ti']
    # Retrieve the anomaly detection results from XCom
    anomaly_results = ti.xcom_pull(task_ids='anomaly_detection')
    # Check if anomaly_results is None before proceeding
    if anomaly_results is None:
        logging.error("No data received from anomaly detection task. Skipping alert.")
        return  # Exit the function if no data is found
    
    issues = anomaly_results.get('issues', {})

    try:
        if issues:  # If any issues were detected, proceed to send an alert
            subject = "Anomaly Detected in Dataset"
            html_content = "<h3>Detected Anomalies:</h3>"
            for issue, description in issues.items():
                html_content += f"<p><strong>{issue}:</strong> {description}</p>"
            
            # Optionally, include general statistics in the alert
            stats = anomaly_results.get('stats', {})
            html_content += "<h4>Statistics:</h4>"
            for stat, value in stats.items():
                html_content += f"<p><strong>{stat}:</strong> {value}</p>"

            # Send the email alert
            send_email(to=send_alert_to, subject=subject, html_content=html_content)
            logging.info("Alert email sent due to detected anomalies.")
            return True
        else:
            logging.info("No anomalies detected, no alert sent.")
    except Exception as e:
        logging.error(f"An error occurred in check_anomalies_and_alert: {e}")

# Task definition in your DAG
alert_task = PythonOperator(
    task_id='send_alert_if_anomalies',
    python_callable=check_anomalies_and_alert,
    provide_context=True,
    dag=dag,
)

email_notification_task = EmailOperator(
    task_id='send_email_notification',
    to='aishwariya.alagesanus@gmail.com',
    subject='Dag Completed Successfully',
    html_content='<p>Dag Completed</p>',
    dag=dag,
)

# Define task dependencies
download_task >> load_task >> stats_validate_task >> anomaly_validate_task >> alert_task >> pre_process_task1 >> pre_process_task2 >> eda_task >> encode_categorical_task >> correlation_analysis_task >> smote_analysis_task >> email_notification_task


if __name__ == "__main__":
    dag.cli()
