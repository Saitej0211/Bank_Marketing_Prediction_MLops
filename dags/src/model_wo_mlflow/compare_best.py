import os
import logging
import json
import pickle

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
            return
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

    except Exception as e:
        logger.error(f"Failed to save the best model or metrics: {e}")

if __name__ == "__main__":
    logger.info("Starting model comparison and selection process.")
    compare_and_select_best()
    logger.info("Model comparison and selection process complete.")