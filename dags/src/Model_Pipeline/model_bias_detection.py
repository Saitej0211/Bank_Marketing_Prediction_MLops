import os
import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Path configurations
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODEL_DIR = os.path.join(PROJECT_DIR, "final_model")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "bias_analysis_output")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(path, sample_size=None):
    """Load data from CSV file with optional sampling"""
    logger.info(f"Loading data from {path}")
    data = pd.read_csv(path)
    if sample_size and len(data) > sample_size:
        data = data.sample(n=sample_size, random_state=42)
    X = data.drop('y', axis=1)
    y = data['y']
    logger.info(f"Data loaded successfully. Shape: {X.shape}")
    return X, y

def slice_data(X, feature, n_bins=5):
    """Slice data based on a specific feature"""
    logger.info(f"Slicing data for feature: {feature}")
    if feature == 'age':
        bins = [-np.inf, 0.2, 0.4, 0.6, 0.8, np.inf]
        labels = ['18-30', '31-45', '46-60', '61-75', '75+']
        return pd.cut(X[feature], bins=bins, labels=labels)
    elif feature == 'marital':
        bins = [-np.inf, 0.33, 0.66, np.inf]
        labels = ['single', 'married', 'divorced']
        return pd.cut(X[feature], bins=bins, labels=labels)
    elif X[feature].dtype == 'object':
        return X[feature].unique()
    else:
        return pd.qcut(X[feature], q=n_bins, duplicates='drop')

def evaluate_slice(model, X, y, feature, slice_value):
    """Evaluate model performance on a specific slice"""
    if feature == 'age':
        if slice_value == '18-30':
            mask = X[feature] <= 0.2
        elif slice_value == '31-45':
            mask = (X[feature] > 0.2) & (X[feature] <= 0.4)
        elif slice_value == '46-60':
            mask = (X[feature] > 0.4) & (X[feature] <= 0.6)
        elif slice_value == '61-75':
            mask = (X[feature] > 0.6) & (X[feature] <= 0.8)
        else:  # 75+
            mask = X[feature] > 0.8
    elif feature == 'marital':
        if slice_value == 'single':
            mask = X[feature] <= 0.33
        elif slice_value == 'married':
            mask = (X[feature] > 0.33) & (X[feature] <= 0.66)
        else:  # divorced
            mask = X[feature] > 0.66
    elif isinstance(slice_value, pd.Interval):
        mask = X[feature].between(slice_value.left, slice_value.right, inclusive='both')
    else:
        mask = X[feature] == slice_value
    
    if mask.sum() == 0:
        logger.warning(f"No samples found for slice {slice_value} of feature {feature}")
        return {
            'slice': str(slice_value),
            'size': 0,
            'accuracy': None,
            'precision': None,
            'recall': None,
            'f1': None
        }
    
    X_slice = X[mask]
    y_slice = y[mask]
    y_pred = model.predict(X_slice)
    accuracy = accuracy_score(y_slice, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_slice, y_pred, average='weighted')
    
    result = {
        'slice': str(slice_value),
        'size': mask.sum(),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    return result

def detect_bias(model, X, y, sensitive_features):
    """Detect bias across sensitive features"""
    logger.info(f"Detecting bias for features: {sensitive_features}")
    bias_results = {}
    for feature in sensitive_features:
        logger.info(f"Analyzing feature: {feature}")
        slices = slice_data(X, feature)
        feature_results = []
        for slice_value in slices.categories if hasattr(slices, 'categories') else slices:
            result = evaluate_slice(model, X, y, feature, slice_value)
            if result['size'] > 0:  # Only add results for non-empty slices
                feature_results.append(result)
        if feature_results:  # Only add feature if it has any non-empty slices
            bias_results[feature] = feature_results
            logger.info(f"Bias detection complete for {feature}. Number of valid slices: {len(feature_results)}")
        else:
            logger.warning(f"No valid slices found for feature {feature}")
    return bias_results

def visualize_bias(bias_results, metric='accuracy', output_dir=OUTPUT_DIR):
    """Visualize bias across features and slices"""
    logger.info("Visualizing bias results")
    for feature, results in bias_results.items():
        plt.figure(figsize=(10, 6))
        sns.barplot(x=[r['slice'] for r in results], y=[r[metric] for r in results])
        plt.title(f'{metric.capitalize()} across {feature} slices')
        plt.xlabel(feature)
        plt.ylabel(metric.capitalize())
        plt.xticks(rotation=45)
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'bias_{feature}_{metric}.png')
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Visualization saved to {output_path}")

def run_bias_analysis(model_path, test_path, sensitive_features, sample_size=1000):
    """Run the entire bias analysis process"""
    logger.info("Starting bias analysis")
    
    # Load the trained model
    logger.info(f"Loading model from: {model_path}")
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info("Model loaded successfully")
    except FileNotFoundError:
        logger.error(f"Model file not found at {model_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

    # Load data
    X, y = load_data(test_path, sample_size)

    # Detect bias
    bias_results = detect_bias(model, X, y, sensitive_features)

    if not bias_results:
        logger.warning("No valid bias results found. Skipping visualization.")
        return {}

    # Visualize bias
    visualize_bias(bias_results)

    logger.info("Bias analysis completed")
    return bias_results

if __name__ == "__main__":
    logger.info("Starting bias analysis script")
    model_path = os.path.join(MODEL_DIR, "random_forest_latest.pkl")
    test_path = os.path.join(DATA_DIR, "processed", "test_data.csv")
    sensitive_features = ['age', 'marital']
    sample_size = 1000  # Adjust this value based on your needs

    bias_results = run_bias_analysis(model_path, test_path, sensitive_features, sample_size)
    logger.info("Bias analysis completed.")
    logger.info(f"Results saved in {OUTPUT_DIR}")