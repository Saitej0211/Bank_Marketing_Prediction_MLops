import os
import json
import logging
import pickle
import mlflow
import mlflow.sklearn

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the directory where models and results are stored
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "final_model")

def load_metrics(file_path):
    """Load model metrics from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            metrics = json.load(f)
        logger.info(f"Loaded metrics from {file_path}")
        return metrics
    except Exception as e:
        logger.error(f"Failed to load metrics from {file_path}: {e}")
        return None

def get_all_model_metrics():
    """Retrieve all model metrics files in DATA_DIR."""
    metrics_files = [f for f in os.listdir(DATA_DIR) if f.startswith("results_") and f.endswith(".json")]
    if not metrics_files:
        logger.warning("No model metrics files found in DATA_DIR.")
    else:
        logger.info(f"Found {len(metrics_files)} metrics files in DATA_DIR.")
    return [os.path.join(DATA_DIR, f) for f in metrics_files]

def simplify_metrics(metrics):
    """Flatten nested dictionary metrics to make them JSON-serializable."""
    simplified_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                simplified_metrics[f"{key}_{sub_key}"] = sub_value  # Flatten nested dict
        else:
            simplified_metrics[key] = value
    return simplified_metrics

def log_best_model_in_mlflow(best_model_path, best_metrics):
    """Log and register the best model and its metrics in MLflow."""
    with mlflow.start_run(run_name="Best_Model_Logging"):
        # Log only numeric metrics
        for metric_name, metric_value in best_metrics.items():
            if isinstance(metric_value, (int, float)):
                mlflow.log_metric(metric_name, metric_value)
            else:
                logger.warning(f"Skipping non-numeric metric {metric_name}: {metric_value}")

        # Log and register the model
        with open(best_model_path, 'rb') as model_file:
            model = pickle.load(model_file)
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="best_model"
            )
            
            # Register the model
            model_name = "best_random_forest_model"
            mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/best_model", model_name)
            
            # Transition the model to the "Staging" stage
            client = mlflow.tracking.MlflowClient()
            latest_version = client.get_latest_versions(model_name, stages=["None"])[0].version
            client.transition_model_version_stage(
                name=model_name,
                version=latest_version,
                stage="Staging"
            )
        
        logger.info(f"Logged and registered the best model '{model_name}' in MLflow and transitioned to 'Staging' stage.")

def get_test_file_paths(best_model_path):
    """
    Identify and return the paths for X_test and y_test files
    based on the best model path.

    Parameters:
    best_model_path (str): Path to the best model file
    """
    test_file_name = os.path.basename(best_model_path).replace(".pkl", "")
    X_test_path = os.path.join(DATA_DIR, f"{test_file_name}_X_test.csv")
    y_test_path = os.path.join(DATA_DIR, f"{test_file_name}_y_test.csv")

    logger.info(f"Identified X_test path: {X_test_path}")
    logger.info(f"Identified y_test path: {y_test_path}")

    return X_test_path, y_test_path

def compare_and_select_best():
    """Compare models from different sessions and update the best model if necessary."""
    all_metrics_files = get_all_model_metrics()
    if not all_metrics_files:
        logger.error("No metrics files available for comparison. Exiting.")
        return

    best_metrics = None
    best_model_path = None

    # Loop through each model's results file and compare metrics
    for metrics_file in all_metrics_files:
        metrics = load_metrics(metrics_file)
        if metrics is None:
            logger.warning(f"Skipping {metrics_file} due to loading issues.")
            continue

        # Update best model if this model's accuracy is higher
        if best_metrics is None or metrics['accuracy'] > best_metrics['accuracy']:
            best_metrics = metrics
            model_file = metrics_file.replace("results_", "random_forest_").replace(".json", ".pkl")
            best_model_path = os.path.join(DATA_DIR, model_file)
            logger.info(f"New best model found with accuracy {metrics['accuracy']} from {metrics_file}")

    # Load previously saved best model metrics, if any
    best_metrics_path = os.path.join(DATA_DIR, "best_model.json")
    if os.path.exists(best_metrics_path):
        previous_best_metrics = load_metrics(best_metrics_path)
        if previous_best_metrics and previous_best_metrics.get('accuracy', 0) >= best_metrics['accuracy']:
            logger.info("Previous best model still performs better. No update made.")
            # Call get_test_file_paths to get X_test and y_test paths
            X_test_path, y_test_path = get_test_file_paths(best_model_path)
            logger.info(f'Compare and select best-- X_test_path: {X_test_path}')
            logger.info(f'y_test_path: {y_test_path}')
            return best_model_path, X_test_path, y_test_path    # Return the path to the previous best model
        else:
            logger.info("Previous best model is being replaced by a new model.")

    # Update the best model and metrics if a new best model is found
    try:
        # Simplify metrics to ensure they are JSON-serializable
        flattened_metrics = simplify_metrics(best_metrics)
        
        # Save metrics to JSON
        with open(best_metrics_path, 'w') as f:
            json.dump(flattened_metrics, f, indent=4)
        logger.info(f"Saved new best model metrics to {best_metrics_path}")

        # Save the model file as the best model
        best_model_file = os.path.join(DATA_DIR, "best_model.pkl")
        with open(best_model_path, 'rb') as src, open(best_model_file, 'wb') as dest:
            dest.write(src.read())
        logger.info(f"Saved new best model file to {best_model_file}")

        # Log best model and metrics in MLflow
        log_best_model_in_mlflow(best_model_file, flattened_metrics)
        
        # Call get_test_file_paths to get X_test and y_test paths
        X_test_path, y_test_path = get_test_file_paths(best_model_path)

        logger.info(f'Compare and select best-- X_test_path: {X_test_path}')
        logger.info(f'y_test_path: {y_test_path}')

         # Return the path to the new best model file
        return best_model_file, X_test_path, y_test_path

    except Exception as e:
        logger.error(f"Failed to save the best model or metrics: {e}")

if __name__ == "__main__":
    logger.info("Starting model comparison and selection process.")
    compare_and_select_best()
    logger.info("Model comparison and selection process complete.")