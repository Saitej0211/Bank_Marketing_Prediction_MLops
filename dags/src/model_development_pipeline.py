from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import os

# Import the function from your script
from src.Model_Pipeline.model_development_with_mlflow import run_model_development

default_args = {
    'owner': 'MLopsProjectGroup6',
    'depends_on_past': False,
    'start_date': datetime(2024, 10, 19),
    'retries': 2,
    'retry_delay': timedelta(minutes=5)
}

dag_2 = DAG(
    'ModelDevelopmentPipeline',
    default_args=default_args,
    description='DAG for running model development with MLflow',
    schedule_interval=None,
    catchup=False,
)

def run_model_development_task():
    TRAIN_PATH = "/opt/airflow/data/processed/smote_resampled_train_data.csv"
    TEST_PATH = "/opt/airflow/data/processed/test_data.csv"
    run_model_development(TRAIN_PATH, TEST_PATH)

model_development_task = PythonOperator(
    task_id='run_model_development',
    python_callable=run_model_development_task,
    dag=dag_2,
)

model_development_task