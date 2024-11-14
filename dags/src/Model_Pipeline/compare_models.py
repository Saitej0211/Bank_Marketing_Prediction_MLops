import os
import json
import glob
import logging
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# File paths
PAR_DIRECTORY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(PAR_DIRECTORY, "models")
FINAL_DIR = os.path.join(MODELS_DIR, "final")

def get_latest_results():
    """Get the two most recent result files, sorted by creation time (newest first)."""
    result_files = sorted(glob.glob(os.path.join(MODELS_DIR, "results_*.json")), key=os.path.getctime, reverse=True)
    return result_files[:2] if len(result_files) >= 2 else (result_files[0], None) if result_files else (None, None)

def load_json(file_path):
    """Load JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def compare_results(current_results, previous_results):
    """Compare current results with the previous results."""
    # You can modify this function to change the comparison criteria
    return current_results['accuracy'] > previous_results['accuracy']

def get_corresponding_pkl(json_path):
    """Get the corresponding PKL file for a given JSON result file."""
    json_name = os.path.basename(json_path)
    pkl_name = json_name.replace('results_', 'random_forest_').replace('.json', '.pkl')
    return os.path.join(MODELS_DIR, pkl_name)

def update_final_model(current_result):
    """Copy the current result JSON and corresponding PKL to the final folder."""
    os.makedirs(FINAL_DIR, exist_ok=True)
    
    # Copy JSON
    final_json_path = os.path.join(FINAL_DIR, "best_model.json")
    shutil.copy2(current_result, final_json_path)
    logger.info(f"Updated best model JSON in final folder: {final_json_path}")
    
    # Copy PKL
    pkl_path = get_corresponding_pkl(current_result)
    if os.path.exists(pkl_path):
        final_pkl_path = os.path.join(FINAL_DIR, "best_model.pkl")
        shutil.copy2(pkl_path, final_pkl_path)
        logger.info(f"Updated best model PKL in final folder: {final_pkl_path}")
    else:
        logger.warning(f"Corresponding PKL file not found for {current_result}")

def is_final_dir_empty():
    """Check if the final directory is empty."""
    return not os.path.exists(FINAL_DIR) or not os.listdir(FINAL_DIR)

def main():
    current_result, previous_result = get_latest_results()

    if not current_result:
        logger.info("No result files found. Nothing to do.")
        return

    if not previous_result or is_final_dir_empty():
        # If there's only one result file or the final directory is empty, use the current model
        update_final_model(current_result)
        return

    current_results = load_json(current_result)
    previous_results = load_json(previous_result)

    if compare_results(current_results, previous_results):
        update_final_model(current_result)
    else:
        if is_final_dir_empty():
            logger.info("Current model is not better, but final directory is empty. Updating with current model.")
            update_final_model(current_result)
        else:
            logger.info("Current model is not better than the previous model. No changes made.")

if __name__ == "__main__":
    main()