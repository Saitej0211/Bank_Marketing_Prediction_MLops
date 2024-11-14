import os
import logging
import json
import pickle
import pandas as pd  # Make sure pandas is imported
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "final_model")

# Hyperparameter search space for faster execution
SPACE = {
    'n_estimators': [50, 100, 150],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False]
}

def load_data(train_path, test_path):
    """Load train and test data from CSV files"""
    try:
        train_data = pd.read_csv(train_path)
        logger.info(f"Loaded train data from {train_path} with shape {train_data.shape}")
        test_data = pd.read_csv(test_path)
        logger.info(f"Loaded test data from {test_path} with shape {test_data.shape}")
        
        # Split features and target variable
        X_train = train_data.drop('y', axis=1)  # Assuming 'y' is the target column
        y_train = train_data['y']
        X_test = test_data.drop('y', axis=1)
        y_test = test_data['y']
        
        return X_train, y_train, X_test, y_test
    except Exception as e:
        logger.exception(f"Error loading data: {e}")
        raise

def evaluate_model_performance(y_test, y_pred):
    """Evaluate model performance"""
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }
    
    logger.info(f"Model performance: {metrics}")
    
    # Return True if performance meets threshold (e.g., accuracy >= 0.7)
    return accuracy >= 0.7

def save_model_and_results(model, results):
    """Save the model and results as JSON in the models folder"""
    models_dir = DATA_DIR
    os.makedirs(models_dir, exist_ok=True)

    # Save model as pickle file
    model_path = os.path.join(models_dir, f"best_random_forest_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    # Save metrics as a JSON file
    results_path = os.path.join(models_dir, f"best_model_metrics.json")
    with open(results_path, 'w') as f:
        json.dump(results, f)

def train_and_evaluate(X_train, y_train, X_test, y_test):
    """Train a model and evaluate its performance"""
    
    best_model = None
    best_metrics = None
    
    for params in SPACE:
        # Train a RandomForest model with current params
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # Predict on test set and evaluate performance
        y_pred = model.predict(X_test)
        performance_ok = evaluate_model_performance(y_test, y_pred)

        if best_model is None or performance_ok:
            best_model = model
            
            # Save metrics for best model so far (you can add more metrics if needed)
            best_metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_recall_fscore_support(y_test)[0],
                "recall": precision_recall_fscore_support(y_test)[1]
            }

            logger.info(f"New best model found with params: {params}")

            # Save the best model and its metrics locally (instead of logging to MLFlow)
            save_model_and_results(best_model, best_metrics)

if __name__ == "__main__":
    
    TRAIN_PATH = os.path.join(PROJECT_DIR,"data", "processed", "train.csv")
    TEST_PATH = os.path.join(PROJECT_DIR,"data", "processed", "test.csv")
    
   # Load data 
X_train,y_train,X_test,y_test=load_data(TRAIN_PATH , TEST_PATH)

   # Train and evaluate models directly without using MLFlow 
train_and_evaluate(X_train,y_train,X_test,y_test)