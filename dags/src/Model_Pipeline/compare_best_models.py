
import os
import json
import logging
import pickle
import warnings
import mlflow
import mlflow.sklearn
import time
from airflow.utils.log.logging_mixin import LoggingMixin
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

# Setup MLflow Tracking URI
mlflow.set_tracking_uri("http://mlflow:5000")  

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# Paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "final_model")
LOG_FILE_PATH = os.path.join(PROJECT_DIR, "dags", "logs", "compare_best_models.log")

# Create necessary directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)

# Custom file logger
file_handler = logging.FileHandler(LOG_FILE_PATH, mode='a')
logger.addHandler(file_handler)

# Model Registry name
MODEL_NAME = "best_random_forest_model"

# Utility functions
def load_json(file_path):
    """Load JSON data from a file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded data from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        return None

def flatten_metrics(metrics):
    """Flatten nested dictionary metrics."""
    return {f"{k}_{sub_k}": sub_v if isinstance(v, dict) else v 
            for k, v in metrics.items() 
            for sub_k, sub_v in (v.items() if isinstance(v, dict) else [(k, v)])}

# Model comparison and selection
def compare_models():
    """Identify and log the best model based on accuracy."""
    metrics_files = [f for f in os.listdir(DATA_DIR) if f.startswith("results_") and f.endswith(".json")]
    if not metrics_files:
        logger.warning("No model metrics files found.")
        return

    best_metrics, best_model_path = None, None
    for metrics_file in metrics_files:
        metrics = load_json(os.path.join(DATA_DIR, metrics_file))
        if metrics and (best_metrics is None or metrics['accuracy'] > best_metrics['accuracy']):
            best_metrics, best_model_path = metrics, metrics_file.replace("results_", "random_forest_").replace(".json", ".pkl")
            logger.info(f"New best model found with accuracy {metrics['accuracy']}")

    if not best_metrics:
        logger.error("No valid metrics files found.")
        return

    # Delay before checking for previous best metrics file
    time.sleep(15)

    # Load previous best metrics
    best_metrics_path = os.path.join(DATA_DIR, "best_model.json")
    previous_best_metrics = load_json(best_metrics_path)
    if previous_best_metrics and previous_best_metrics.get('accuracy', 0) >= best_metrics['accuracy']:
        logger.info("Previous best model is still the best.")
        return

    # Save new best model and metrics, and register in MLflow
    save_best_model_and_metrics(best_model_path, flatten_metrics(best_metrics))

def save_best_model_and_metrics(model_path, metrics):
    """Save the best model and its metrics, and log them in MLflow."""
    try:
        # Save metrics
        with open(os.path.join(DATA_DIR, "best_model.json"), 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info("Saved new best metrics.")

        # Save model
        with open(os.path.join(DATA_DIR, "best_model.pkl"), 'wb') as dest, open(os.path.join(DATA_DIR, model_path), 'rb') as src:
            dest.write(src.read())
        logger.info("Saved new best model.")

        # Log in MLflow and register the model
        with mlflow.start_run(run_name="Best_Model_Logging") as run:
            logger.info(f"Started MLflow run with ID: {run.info.run_id}")
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(metric_name, metric_value)

            # Log the model as an artifact in MLflow
            mlflow.sklearn.log_model(
                pickle.load(open(os.path.join(DATA_DIR, "best_model.pkl"), 'rb')),
                artifact_path="best_model"
            )
            logger.info("Logged best model to MLflow.")

            # Register the model in the MLflow Model Registry
            model_uri = f"runs:/{run.info.run_id}/best_model"
            logger.info(f"Model URI: {model_uri}")
            client = MlflowClient()

            # Try to create the model in the registry, if it doesn't already exist
            try:
                client.create_registered_model(MODEL_NAME)
                logger.info(f"Created registered model {MODEL_NAME}")
            except MlflowException:
                logger.info(f"Model {MODEL_NAME} already exists in registry.")

            # Register a new version of the model
            model_version = client.create_model_version(
                name=MODEL_NAME,
                source=model_uri,
                run_id=run.info.run_id
            )
            logger.info(f"Registered model version: {model_version.version}")

            # Transition model version to "Staging"
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=model_version.version,
                stage="Staging"
            )
            logger.info(f"Transitioned model {MODEL_NAME} version {model_version.version} to 'Staging'")

    except Exception as e:
        logger.error(f"Failed to save or log best model: {e}")

# Run
if __name__ == "__main__":
    logger.info("Starting model comparison and selection.")
    compare_models()
    logger.info("Process complete.")
