from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.email import EmailOperator
from airflow.utils.email import send_email
from datetime import datetime, timedelta
import os
import logging

# Import the functions from your scripts
from src.Model_Pipeline.model_bias_detection import run_bias_analysis
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

def notify_success(context):
    dag_run = context['dag_run']
    msg = f"DAG {dag_run.dag_id} has completed successfully."
    subject = f"Success: {dag_run.dag_id}"
    send_email(to='hashwanthmoorthy@gmail.com', subject=subject, html_content=msg)

def notify_failure(context):
    dag_run = context['dag_run']
    task = context['task']
    msg = f"Task {task.task_id} in DAG {dag_run.dag_id} failed."
    subject = f"Failure: {dag_run.dag_id} - {task.task_id}"
    send_email(to='hashwanthmoorthy@gmail.com', subject=subject, html_content=msg)

dag_2 = DAG(
    'ModelDevelopmentPipeline',
    default_args=default_args,
    description='DAG for running model development with MLflow',
    schedule_interval=None,
    catchup=False,
    is_paused_upon_creation=False,
    on_success_callback=notify_success,
    on_failure_callback=notify_failure
)

def run_model_development_task(**kwargs):
    TRAIN_PATH = "files/md5/75/Processed_Files/smote_resampled_train_data"
    TEST_PATH = "files/md5/75/Processed_Files/test_data"
    
    final_metrics = run_model_development(TRAIN_PATH, TEST_PATH, max_attempts=3)
    
    logging.info(f"Final model metrics: {final_metrics}")
    kwargs['ti'].xcom_push(key='final_metrics', value=final_metrics)
    
    if all(metric >= 0.7 for metric in final_metrics.values()):
        logging.info("Model development successful: All metrics are above 0.7")
    else:
        logging.warning("Model development completed, but not all metrics are above 0.7")

model_development_task = PythonOperator(
    task_id='run_model_development',
    python_callable=run_model_development_task,
    provide_context=True,
    on_failure_callback=notify_failure,
    dag=dag_2,
)

def find_best_model(**kwargs):
    best_model_info = compare_and_select_best()
    if best_model_info:
        best_model_path, X_test_path, y_test_path = best_model_info
        logging.info(f"Best model path: {best_model_path}")
        logging.info(f"X_test path: {X_test_path}")
        logging.info(f"y_test path: {y_test_path}")
        kwargs['ti'].xcom_push(key='best_model_path', value=best_model_path)
        kwargs['ti'].xcom_push(key='X_test_path', value=X_test_path)
        kwargs['ti'].xcom_push(key='y_test_path', value=y_test_path)
    else:
        logging.error("Failed to find the best model paths.")

compare_best_model_task = PythonOperator(
    task_id='compare_best_models',
    python_callable=find_best_model,
    provide_context=True,
    on_failure_callback=notify_failure,
    dag=dag_2,
)

sensitivity_analysis_task = PythonOperator(
    task_id="sensitivity_analysis_task", 
    python_callable=perform_sensitivity_analysis,
    op_args=[
        "{{ ti.xcom_pull(task_ids='compare_best_models', key='best_model_path') }}",
        "{{ ti.xcom_pull(task_ids='compare_best_models', key='X_test_path') }}",
        "{{ ti.xcom_pull(task_ids='compare_best_models', key='y_test_path') }}"
    ],
    on_failure_callback=notify_failure,
    dag=dag_2
)

def run_bias_analysis_task(**kwargs):
    ti = kwargs['ti']
    model_path = ti.xcom_pull(task_ids='compare_best_models', key='best_model_path')
    if not model_path:
        logging.error("Model path not found in XCom. Cannot proceed with bias analysis.")
        return
    TEST_PATH = "/opt/airflow/data/processed/test_data.csv"
    sensitive_features = ['age', 'marital']
    sample_size = 1000
    try:
        bias_results = run_bias_analysis(model_path, TEST_PATH, sensitive_features, sample_size)
        if bias_results:
            for feature, results in bias_results.items():
                for result in results:
                    logging.info(f"Feature: {feature}, Slice: {result['slice']}, Accuracy: {result['accuracy']:.4f}")
            ti.xcom_push(key='bias_results', value=bias_results)
        else:
            logging.warning("No bias results found.")
    except Exception as e:
        logging.error(f"Error during bias analysis: {str(e)}")
        raise

bias_analysis_task = PythonOperator(
    task_id='run_bias_analysis',
    python_callable=run_bias_analysis_task,
    provide_context=True,
    on_failure_callback=notify_failure,
    dag=dag_2,
)

push_to_gcp_task = PythonOperator(
    task_id='push_to_gcp',
    python_callable=push_to_gcp,
    on_failure_callback=notify_failure,
    dag=dag_2,
)

email_notification_task = EmailOperator(
    task_id='send_completion_email',
    to='hashwanthmoorthy@gmail.com',
    subject='Model Development Pipeline Completed Successfully',
    html_content='<p>The Model Development Pipeline has completed successfully.</p>',
    dag=dag_2,
)

# Set the task dependencies
model_development_task >> compare_best_model_task >> sensitivity_analysis_task >> bias_analysis_task >> push_to_gcp_task >> email_notification_task

if __name__ == "__main__":
    dag_2.cli()