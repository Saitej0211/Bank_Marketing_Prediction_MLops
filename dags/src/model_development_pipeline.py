from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import os
import logging

# Import the function from your script
from src.Model_Pipeline.model_development_and_evaluation_with_mlflow import run_model_development
from src.Model_Pipeline.compare_best_models import compare_and_select_best

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

def run_model_development_task(**kwargs):
    TRAIN_PATH = "/opt/airflow/data/processed/smote_resampled_train_data.csv"
    TEST_PATH = "/opt/airflow/data/processed/test_data.csv"
    
    # Run the model development process
    final_metrics = run_model_development(TRAIN_PATH, TEST_PATH, max_attempts=3)
    
    # Log the final metrics
    logging.info(f"Final model metrics: {final_metrics}")
    
    # You can also push the metrics to XCom if you want to use them in downstream tasks
    kwargs['ti'].xcom_push(key='final_metrics', value=final_metrics)
    
    # Check if the model met the performance threshold
    if all(metric >= 0.7 for metric in final_metrics.values()):
        logging.info("Model development successful: All metrics are above 0.7")
    else:
        logging.warning("Model development completed, but not all metrics are above 0.7")

model_development_task = PythonOperator(
    task_id='run_model_development',
    python_callable=run_model_development_task,
    provide_context=True,  # This allows the function to receive kwargs
    dag=dag_2,
)

# Define the task for comparing best models
compare_best_model_task = PythonOperator(
    task_id='compare_best_models',
    python_callable=compare_and_select_best,
    dag=dag_2,
)

# Set the task dependencies
model_development_task >> compare_best_model_task