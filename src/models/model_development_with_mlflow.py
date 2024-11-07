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
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from hyperopt.pyll import scope
import pickle

# Loading file paths for CSVs
PAR_DIRECTORY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
__TRAINPATH__ = os.path.join(PAR_DIRECTORY, "data", "processed", "smote_resampled_train_data.csv")
__TESTPATH__ = os.path.join(PAR_DIRECTORY, "data", "processed", "test_data.csv")
MODELS_DIR = os.path.join(PAR_DIRECTORY, "models")

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
mlflow.set_tracking_uri("http://127.0.0.1:5001")
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
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Save model
    model_path = os.path.join(MODELS_DIR, f"random_forest_{run_name}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save results
    results['timestamp'] = run_name
    results_path = os.path.join(MODELS_DIR, f"results_{run_name}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Results saved to {results_path}")

def train_and_log_model(X_train, y_train, X_test, y_test):
    """Train multiple models and log results with MLflow"""
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    client = MlflowClient()
    experiment = client.get_experiment_by_name("random_forest_classification")
    
    with mlflow.start_run(run_name=run_name, experiment_id=experiment.experiment_id) as parent_run:
        mlflow.set_tag("run_name", run_name)
        
        trials = Trials()
        fmin_objective = lambda params: objective(params, X_train, y_train)
        best = fmin(fn=fmin_objective, space=SPACE, algo=tpe.suggest, max_evals=10, trials=trials)
        
        for i, trial in enumerate(trials.trials):
            with mlflow.start_run(run_name=f"{run_name}_model_{i}", nested=True) as child_run:
                params = {
                    'n_estimators': int(trial['misc']['vals']['n_estimators'][0]),
                    'max_depth': int(trial['misc']['vals']['max_depth'][0]),
                    'min_samples_split': int(trial['misc']['vals']['min_samples_split'][0]),
                    'min_samples_leaf': int(trial['misc']['vals']['min_samples_leaf'][0]),
                    'max_features': ['sqrt', 'log2'][trial['misc']['vals']['max_features'][0]],
                    'bootstrap': [True, False][trial['misc']['vals']['bootstrap'][0]]
                }
                
                model = RandomForestClassifier(**params, n_jobs=-1)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
                
                mlflow.log_params(params)
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1", f1)
                
                signature = infer_signature(X_test, y_pred)
                mlflow.sklearn.log_model(model, f"model_{i}", signature=signature)
                
                logger.info(f"Logged model {i} with run ID: {child_run.info.run_id}")
        
        # Get the best model
        best_run = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{parent_run.info.run_id}'",
            order_by=["metrics.accuracy DESC"],
            max_results=1
        )[0]
        
        best_model = mlflow.sklearn.load_model(f"runs:/{best_run.info.run_id}/model_{best_run.data.tags['mlflow.runName'].split('_')[-1]}")
        
        # Prepare results for the best model
        results = {
            "accuracy": best_run.data.metrics["accuracy"],
            "precision": best_run.data.metrics["precision"],
            "recall": best_run.data.metrics["recall"],
            "f1_score": best_run.data.metrics["f1"],
            "best_params": best_run.data.params,
            "run_id": best_run.info.run_id
        }
        
        # Save the best model and results locally
        save_model_and_results(best_model, results, run_name)
        
        logger.info(f"Best model run ID: {best_run.info.run_id}")
        return best_run.info.run_id

def main():
    """Main function to load data and run the training process"""
    X_train, y_train, X_test, y_test = load_data(__TRAINPATH__, __TESTPATH__)
    run_id = train_and_log_model(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()  