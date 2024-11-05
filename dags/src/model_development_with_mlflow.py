""" MLflow module for Random Forest model training and logging artifacts. """
import os
import logging
import warnings
import json
import io
import pickle
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from hyperopt.pyll import scope

# Loading file
PAR_DIRECTORY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
__TRAINPATH__ = os.path.join(PAR_DIRECTORY, "data", "processed", "smote_resampled_train_data.pkl")
__TESTPATH__ = os.path.join(PAR_DIRECTORY, "data", "processed", "test_data.pkl")

# Reduced hyperparameter search space for faster execution
SPACE = {
    'n_estimators': scope.int(hp.quniform('n_estimators', 50, 150, 50)),
    'max_depth': scope.int(hp.quniform('max_depth', 5, 15, 5)),
    'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 6, 2)),
    'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 3, 1)),
    'max_features': hp.choice('max_features', ['sqrt', 'log2']),
    'bootstrap': hp.choice('bootstrap', [True, False])
}

# Logging Declarations
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("random_forest_classification")

def load_data(train_path, test_path):
    """Load train and test data from pickle files"""
    try:
        # Load train data
        with open(train_path, 'rb') as f:
            train_loaded_data = pickle.load(f)
        
        # Load test data
        with open(test_path, 'rb') as f:
            test_loaded_data = pickle.load(f)
        
        # Debug print statements
        logger.info(f"Train data type: {type(train_loaded_data)}")
        logger.info(f"Test data type: {type(test_loaded_data)}")
        
        # Function to convert loaded data to DataFrame
        def to_dataframe(data):
            if isinstance(data, pd.DataFrame):
                return data
            elif isinstance(data, bytes):
                return pd.read_csv(io.StringIO(data.decode('utf-8')), sep=';')
            elif isinstance(data, str):
                return pd.read_csv(io.StringIO(data), sep=';')
            else:
                raise ValueError(f"Unexpected data type: {type(data)}")

        train_data = to_dataframe(train_loaded_data)
        test_data = to_dataframe(test_loaded_data)

        logger.info("Train data head:")
        logger.info(train_data.head())
        logger.info("Test data head:")
        logger.info(test_data.head())
        
        X_train = train_data.drop('y', axis=1)
        y_train = train_data['y']
        X_test = test_data.drop('y', axis=1)
        y_test = test_data['y']
        
        return X_train, y_train, X_test, y_test
    except FileNotFoundError as e:
        logger.exception(f"Pickle file not found. Error: {e}")
        raise
    except pd.errors.EmptyDataError as e:
        logger.exception(f"The CSV data in the pickle file is empty. Error: {e}")
        raise
    except Exception as e:
        logger.exception(f"Unable to load files. Error: {e}")
        raise

def objective(params, X, y):
    """Objective function for hyperopt to minimize"""
    clf = RandomForestClassifier(**params, n_jobs=-1)
    score = cross_val_score(clf, X, y, cv=3, scoring='accuracy', n_jobs=-1).mean()
    return {'loss': -score, 'status': STATUS_OK}

def compare_models(new_model, old_model, X_test, y_test):
    new_accuracy = accuracy_score(y_test, new_model.predict(X_test))
    old_accuracy = accuracy_score(y_test, old_model.predict(X_test))
    return new_accuracy > old_accuracy

def notify(message):
    # Implement notification logic (e.g., send email or Slack message)
    logger.info(f"Notification: {message}")

def push_to_registry(model, model_name):
    # Implement logic to push model to a registry
    logger.info(f"Pushed model {model_name} to registry")

def train_and_log_model(X_train, y_train, X_test, y_test):
    """Train the model and log results with MLflow"""
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("run_name", run_name)

        # Hyperparameter tuning
        trials = Trials()
        fmin_objective = lambda params: objective(params, X_train, y_train)
        best = fmin(fn=fmin_objective,
                    space=SPACE,
                    algo=tpe.suggest,
                    max_evals=10,
                    trials=trials)

        # Create best model
        best_params = {
            'n_estimators': int(best['n_estimators']),
            'max_depth': int(best['max_depth']),
            'min_samples_split': int(best['min_samples_split']),
            'min_samples_leaf': int(best['min_samples_leaf']),
            'max_features': ['sqrt', 'log2'][best['max_features']],
            'bootstrap': [True, False][best['bootstrap']]
        }

        model = RandomForestClassifier(**best_params, n_jobs=-1)
        model.fit(X_train, y_train)

        # Make predictions and calculate metrics
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

        # Log parameters
        mlflow.log_params(best_params)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

        # Log the model
        signature = infer_signature(X_test, y_pred)
        mlflow.sklearn.log_model(model, "model", signature=signature)

        # Compare with previous best model
        try:
            old_model = mlflow.sklearn.load_model("models:/RandomForestClassifier/Production")
            if compare_models(model, old_model, X_test, y_test):
                push_to_registry(model, "RandomForestClassifier")
                notify("New model outperforms old model and has been pushed to registry")
            else:
                notify("New model does not outperform old model. Rolling back.")
        except Exception as e:
            logger.error(f"Error comparing models: {e}")
            push_to_registry(model, "RandomForestClassifier")
            notify("First model pushed to registry")

        # Log run details
        run_id = run.info.run_id
        logger.info(f"Run ID: {run_id}")
        
        return run_id, run_name

def main():
    """Main function to run the model training process"""
    X_train, y_train, X_test, y_test = load_data(__TRAINPATH__, __TESTPATH__)
    
    run_dict = {}
    run_id, run_name = train_and_log_model(X_train, y_train, X_test, y_test)
    run_dict[run_name] = run_id

    # Save run details
    p = os.path.join(PAR_DIRECTORY, datetime.now().strftime("%Y%m%d"))
    file = os.path.join(p, datetime.now().strftime("%H%M%S") + ".json")
    if not os.path.exists(p):
        os.makedirs(p)

    with open(file, "w") as f:
        json.dump(run_dict, f, indent=6)

if __name__ == "__main__":
    main()