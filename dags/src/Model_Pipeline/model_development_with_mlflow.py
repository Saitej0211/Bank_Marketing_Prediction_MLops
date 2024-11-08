import os
import logging
import warnings
import json
from datetime import datetime
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from hyperopt.pyll import scope
import pickle 

# Logging Declarations
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "final_model")

# Hyperparameter search space for faster execution
SPACE = {
    'n_estimators': scope.int(hp.quniform('n_estimators', 50, 150, 50)),
    'max_depth': scope.int(hp.quniform('max_depth', 5, 15, 5)),
    'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 6, 2)),
    'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 3, 1)),
    'max_features': hp.choice('max_features', ['sqrt', 'log2']),
    'bootstrap': hp.choice('bootstrap', [True, False])
}

def setup_mlflow():
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("random_forest_classification")

def load_data(train_path, test_path):
    """Load train and test data from CSV files"""
    try:
        train_data = pd.read_csv(train_path)
        logger.info(f"Loaded train data from {train_path} with shape {train_data.shape}")
        test_data = pd.read_csv(test_path)
        logger.info(f"Loaded test data from {test_path} with shape {test_data.shape}")
        
        X_train = train_data.drop('y', axis=1)
        y_train = train_data['y']
        X_test = test_data.drop('y', axis=1)
        y_test = test_data['y']
        return X_train, y_train, X_test, y_test
    except Exception as e:
        logger.exception(f"Error loading data: {e}")
        raise

def objective(params, X, y):
    """Objective function for hyperopt to minimize"""
    clf = RandomForestClassifier(**params, n_jobs=-1)
    score = cross_val_score(clf, X, y, cv=3, scoring='accuracy', n_jobs=-1).mean()
    return {'loss': -score, 'status': STATUS_OK}

def save_model_and_results(model, results, run_name):
    """Save the model and results as JSON in the models folder."""
    models_dir = DATA_DIR
    os.makedirs(models_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(models_dir, f"random_forest_{run_name}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save results
    results['timestamp'] = run_name
    results_path = os.path.join(models_dir, f"results_{run_name}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Results saved to {results_path}")

def train_and_log_model(X_train, y_train, X_test, y_test):
    """Train multiple models and log results with MLflow"""
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    with mlflow.start_run(run_name=run_name) as parent_run:
        mlflow.set_tag("run_name", run_name)

        trials = Trials()
        
        # Perform hyperparameter optimization using Hyperopt
        best_params = fmin(fn=lambda params: objective(params, X_train, y_train), 
                           space=SPACE,
                           algo=tpe.suggest,
                           max_evals=10,
                           trials=trials)

        # Train and log each model from the trials
        for i in range(len(trials.trials)):
            trial = trials.trials[i]
            params = {
                'n_estimators': int(trial['misc']['vals']['n_estimators'][0]),
                'max_depth': int(trial['misc']['vals']['max_depth'][0]),
                'min_samples_split': int(trial['misc']['vals']['min_samples_split'][0]),
                'min_samples_leaf': int(trial['misc']['vals']['min_samples_leaf'][0]),
                'max_features': ['sqrt', 'log2'][trial['misc']['vals']['max_features'][0]],
                'bootstrap': [True, False][trial['misc']['vals']['bootstrap'][0]]
            }

            # Start a nested run for each model training
            with mlflow.start_run(run_name=f"{run_name}_model_{i}", nested=True) as child_run:
                # Train the model with current parameters
                model = RandomForestClassifier(**params)
                model.fit(X_train, y_train)

                # Evaluate on test set
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

                # Log parameters and metrics for this model in MLflow directly without suffixes
                mlflow.log_params(params)
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1_score", f1_score)

                # Log the model to MLflow with a unique name based on trial index
                signature = infer_signature(X_test, y_pred)
                mlflow.sklearn.log_model(model, f"model_{i}", signature=signature)

                logger.info(f"Logged model {i} with parameters: {params}")

                # Prepare results for saving after all models are logged (optional step if needed)
                results = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1_score,
                    "best_params": params,
                    "run_id": child_run.info.run_id,
                    "timestamp": run_name
                }

                # Save the trained model and results locally after logging metrics (optional step if needed)
                save_model_and_results(model, results, run_name)

def run_model_development(train_path, test_path):
    """Function to run the entire model development process"""
    setup_mlflow()
    X_train, y_train, X_test, y_test = load_data(train_path, test_path)
    train_and_log_model(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    # Example usage 
    PAR_DIRECTORY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    TRAIN_PATH = os.path.join(PAR_DIRECTORY, "data", "processed", "smote_resampled_train_data.csv")
    TEST_PATH = os.path.join(PAR_DIRECTORY, "data", "processed", "test_data.csv")
    run_model_development(TRAIN_PATH, TEST_PATH)