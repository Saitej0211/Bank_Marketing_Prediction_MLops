import os
import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
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
    logger.info(f"Loading data from {path}")
    data = pd.read_csv(path)
    if sample_size and len(data) > sample_size:
        data = data.sample(n=sample_size, random_state=42)
    X = data.drop('y', axis=1)
    y = data['y']
    logger.info(f"Data loaded successfully. Shape: {X.shape}")
    return X, y

def slice_data(X, feature, n_bins=5):
    logger.info(f"Slicing data for feature: {feature}")
    if feature == 'age':
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
        labels = ['18-30', '31-45', '46-60', '61-75', '75+']
        return pd.cut(X[feature], bins=bins, labels=labels)
    elif feature == 'marital':
        bins = [0, 0.33, 0.66, 1]
        labels = ['single', 'married', 'divorced']
        return pd.cut(X[feature], bins=bins, labels=labels)
    elif X[feature].dtype == 'object':
        return X[feature]
    else:
        return pd.qcut(X[feature], q=n_bins, duplicates='drop')

def evaluate_slice(model, X, y, feature, slice_value):
    if feature == 'age':
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
        mask = pd.cut(X[feature], bins=bins, labels=['18-30', '31-45', '46-60', '61-75', '75+']) == slice_value
    elif feature == 'marital':
        bins = [0, 0.33, 0.66, 1]
        mask = pd.cut(X[feature], bins=bins, labels=['single', 'married', 'divorced']) == slice_value
    elif isinstance(slice_value, pd.Interval):
        mask = X[feature].between(slice_value.left, slice_value.right, inclusive='both')
    else:
        mask = X[feature] == slice_value
    
    if mask.sum() == 0:
        logger.warning(f"No samples found for slice {slice_value} of feature {feature}")
        return {'slice': str(slice_value), 'size': 0, 'accuracy': None, 'precision': None, 'recall': None, 'f1': None}
    
    X_slice = X[mask]
    y_slice = y[mask]
    y_pred = model.predict(X_slice)
    accuracy = accuracy_score(y_slice, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_slice, y_pred, average='weighted')
    
    return {
        'slice': str(slice_value),
        'size': mask.sum(),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def detect_bias(model, X, y, sensitive_features):
    logger.info(f"Detecting bias for features: {sensitive_features}")
    bias_results = {}
    for feature in sensitive_features:
        logger.info(f"Analyzing feature: {feature}")
        slices = slice_data(X, feature)
        feature_results = []
        for slice_value in slices.categories if hasattr(slices, 'categories') else slices.unique():
            result = evaluate_slice(model, X, y, feature, slice_value)
            if result['size'] > 0:
                feature_results.append(result)
        if feature_results:
            bias_results[feature] = feature_results
            logger.info(f"Bias detection complete for {feature}. Number of valid slices: {len(feature_results)}")
        else:
            logger.warning(f"No valid slices found for feature {feature}")
    return bias_results

def visualize_bias(bias_results, metric='accuracy', output_dir=OUTPUT_DIR):
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

def mitigate_bias(X, y, sensitive_features):
    logger.info("Applying bias mitigation techniques")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply SMOTE for oversampling
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    
    logger.info(f"Data shape after mitigation: {X_resampled.shape}")
    return X_resampled, y_resampled

def run_bias_analysis(model_path, test_path, sensitive_features, sample_size=1000):
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

    # Detect initial bias
    initial_bias_results = detect_bias(model, X, y, sensitive_features)

    if not initial_bias_results:
        logger.warning("No valid bias results found. Skipping visualization.")
        return {}

    # Visualize initial bias
    visualize_bias(initial_bias_results)

    # Mitigate bias
    X_mitigated, y_mitigated = mitigate_bias(X, y, sensitive_features)

    # Train a new model on mitigated data
    mitigated_model = RandomForestClassifier(n_estimators=100, random_state=42)
    mitigated_model.fit(X_mitigated, y_mitigated)

    # Detect bias after mitigation
    mitigated_bias_results = detect_bias(mitigated_model, X, y, sensitive_features)

    # Visualize mitigated bias
    visualize_bias(mitigated_bias_results, output_dir=os.path.join(OUTPUT_DIR, "mitigated"))

    logger.info("Bias analysis and mitigation completed")
    return initial_bias_results, mitigated_bias_results

if __name__ == "__main__":
    logger.info("Starting bias analysis script")
    model_path = os.path.join(MODEL_DIR, "random_forest_latest.pkl")
    test_path = os.path.join(DATA_DIR, "processed", "test_data.csv")
    sensitive_features = ['age', 'marital']
    sample_size = 1000

    initial_results, mitigated_results = run_bias_analysis(model_path, test_path, sensitive_features, sample_size)
    
    logger.info("Bias analysis completed.")
    logger.info(f"Results saved in {OUTPUT_DIR}")

    # Document bias mitigation steps and trade-offs
    with open(os.path.join(OUTPUT_DIR, "bias_mitigation_report.txt"), "w") as f:
        f.write("Bias Mitigation Report\n")
        f.write("======================\n\n")
        f.write("Steps taken to detect and address bias:\n")
        f.write("1. Performed data slicing on sensitive features: age and marital status\n")
        f.write("2. Evaluated model performance across slices using accuracy, precision, recall, and F1-score\n")
        f.write("3. Visualized bias results for initial model\n")
        f.write("4. Applied bias mitigation techniques:\n")
        f.write("   a. Standardized features using StandardScaler\n")
        f.write("   b. Applied SMOTE for oversampling to balance classes\n")
        f.write("5. Trained a new model on the mitigated data\n")
        f.write("6. Re-evaluated and visualized bias results for the mitigated model\n\n")
        f.write("Trade-offs:\n")
        f.write("- Oversampling with SMOTE may introduce synthetic data points, potentially affecting model generalization\n")
        f.write("- Standardization may impact feature interpretability but improves model performance\n")
        f.write("- The mitigated model may have slightly lower overall performance but should exhibit reduced bias across sensitive features\n")

    logger.info("Bias mitigation report generated")