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
from google.cloud import storage



# Loading file paths for CSVs instead of pickles now
PAR_DIRECTORY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
__TRAINPATH__ = os.path.join(PAR_DIRECTORY, "data", "processed", "smote_resampled_train_data.csv")
__TESTPATH__ = os.path.join(PAR_DIRECTORY, "data", "processed", "test_data.csv")
BUCKET_NAME = "mlopsprojectbucketgrp6"

# Set Google Cloud credentials (update the path to your service account key)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(PAR_DIRECTORY, "config", "Key.json")


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
    """Load train and test data from CSV files"""
    try:
        # Load train data from CSV
        train_data = pd.read_csv(train_path)
        logger.info(f"Loaded train data from {train_path} with shape {train_data.shape}")

        # Load test data from CSV
        test_data = pd.read_csv(test_path)
        logger.info(f"Loaded test data from {test_path} with shape {test_data.shape}")

        # Split features (X) and target (y)
        X_train = train_data.drop('y', axis=1)  # Assuming 'y' is the target column in your dataset
        y_train = train_data['y']
        X_test = test_data.drop('y', axis=1)
        y_test = test_data['y']
        
        return X_train, y_train, X_test, y_test
    
    except FileNotFoundError as e:
        logger.exception(f"CSV file not found. Error: {e}")
        raise
    except pd.errors.EmptyDataError as e:
        logger.exception(f"The CSV file is empty. Error: {e}")
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

def upload_model_to_gcs(local_model_path, bucket_name, blob_name):
    """Upload a model to Google Cloud Storage"""
    try:
        # Initialize GCS client
        client = storage.Client()
        
        # Get the bucket
        bucket = client.get_bucket(bucket_name)
        
        # Create a new blob and upload the file's content
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(local_model_path)
        
        logger.info(f"Model uploaded to gs://{bucket_name}/{blob_name}")
    except Exception as e:
        logger.error(f"Error uploading model to GCS: {e}")
        raise

def get_latest_model_version(bucket_name):
    """Retrieve the latest model version from GCS"""
    client = storage.Client()
    blobs = client.list_blobs(bucket_name)

    latest_blob = None

    for blob in blobs:
        if latest_blob is None or blob.updated > latest_blob.updated:
            latest_blob = blob

    if latest_blob:
        return latest_blob.name  # Return the name of the latest blob (model file)
    
    logger.warning("No models found in GCS bucket.")
    return None

def train_and_log_model(X_train, y_train, X_test, y_test):
    """Train the model and log results with MLflow"""
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("run_name", run_name)

        # Hyperparameter tuning using Hyperopt
        trials = Trials()
        fmin_objective = lambda params: objective(params, X_train, y_train)
        
        best = fmin(fn=fmin_objective,
                    space=SPACE,
                    algo=tpe.suggest,
                    max_evals=10,
                    trials=trials)

        # Create best model based on tuned parameters
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

        # Make predictions and calculate metrics on test set
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

        # Log parameters and metrics in MLflow
        mlflow.log_params(best_params)
        mlflow.log_metric("accuracy", accuracy)
        
        # Log additional metrics if needed.
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        # Log the trained model in MLflow with signature for inference later.
        signature = infer_signature(X_test,y_pred)
        local_model_path = f"model_{run_name}.pkl"
        mlflow.sklearn.save_model(model, local_model_path)
        # Define a unique name for the model version using timestamp or incrementing logic.
        blob_name = f"models/random_forest_{run_name}.pkl"
        bucket_name = BUCKET_NAME
        upload_model_to_gcs(local_model_path, bucket_name, blob_name)
        # Compare with previous best model in registry (if exists) and push if better.
        try:
            old_model_blob_name = get_latest_model_version(bucket_name)  # Get latest model from GCS
           
            if old_model_blob_name:
                old_model_local_path = f"old_{run_name}.pkl"
                storage.Blob(old_model_blob_name).download_to_filename(old_model_local_path)  # Download old model
               
                old_model=mlflow.sklearn.load_model(old_model_local_path)  # Load old model
               
                if compare_models(model,old_model,X_test,y_test):
                    push_to_registry(model,"RandomForestClassifier")
                    notify("New model outperforms old model and has been pushed to registry")
                else:
                    notify("New model does not outperform old model. Rolling back.")
            else:
                push_to_registry(model,"RandomForestClassifier")
                notify("First model pushed to registry")
                
        except Exception as e:
            logger.error(f"Error comparing models: {e}")
            push_to_registry(model,"RandomForestClassifier")
            notify("First model pushed to registry")
        # Log run details in MLflow.
        run_id=run.info.run_id
        logger.info(f"Run ID: {run_id}")
        
        return run_id

def main():
    """Main function to load data and run the training process"""
    
    X_train,y_train,X_test,y_test=load_data(__TRAINPATH__,__TESTPATH__)

    run_dict={}
    
    # Train and log the model
    run_id=train_and_log_model(X_train,y_train,X_test,y_test)

if __name__ == "__main__":
     main()