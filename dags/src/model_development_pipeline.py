from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import os
import logging

# Import the function from your script
from src.Model_Pipeline.sensitivity_analysis import perform_sensitivity_analysis
from src.Model_Pipeline.model_development_and_evaluation_with_mlflow import run_model_development
from src.Model_Pipeline.compare_best_models import compare_and_select_best
from src.Model_Pipeline.push_to_gcp import push_to_gcp

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
    is_paused_upon_creation=False 
)

def run_model_development_task(**kwargs):
    TRAIN_PATH = "files/md5/75/Processed_Files/smote_resampled_train_data"
    TEST_PATH = "files/md5/75/Processed_Files/test_data"
    
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

def find_best_model(**kwargs):
    # Perform model comparison and selection logic
    best_model_info = compare_and_select_best()  # Retrieve best model, X_test, and y_test paths

    if best_model_info:
        best_model_path, X_test_path, y_test_path = best_model_info
        logging.info(f"Best model path: {best_model_path}")
        logging.info(f"X_test path: {X_test_path}")
        logging.info(f"y_test path: {y_test_path}")

        # Push the best model paths to XCom for the downstream task
        kwargs['ti'].xcom_push(key='best_model_path', value=best_model_path)
        kwargs['ti'].xcom_push(key='X_test_path', value=X_test_path)
        kwargs['ti'].xcom_push(key='y_test_path', value=y_test_path)
    else:
        logging.error("Failed to find the best model paths.")

compare_best_model_task = PythonOperator(
    task_id='compare_best_models',
    python_callable=find_best_model,
    provide_context=True,  # Enables XCom push
    dag=dag_2,
)

#Sensitivity Analysis
sensitivity_analysis_task = PythonOperator(
    task_id="sensitivity_analysis_task",
    python_callable=perform_sensitivity_analysis,
    op_args=[
        "{{ ti.xcom_pull(task_ids='compare_best_models', key='best_model_path') }}",  # Best model path
        "{{ ti.xcom_pull(task_ids='compare_best_models', key='X_test_path') }}",  # X_test path
        "{{ ti.xcom_pull(task_ids='compare_best_models', key='y_test_path') }}"  # y_test path
    ],
    dag=dag_2
)

# Define the task for pushing the changes to gcp
push_to_gcp_task = PythonOperator(
    task_id='push_to_gcp',
    python_callable=push_to_gcp,
    dag=dag_2,
)

# Set the task dependencies
model_development_task >> compare_best_model_task >> sensitivity_analysis_task >> push_to_gcp_task