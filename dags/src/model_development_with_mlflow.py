import os
import logging
import warnings
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

# Loading file paths for CSVs
PAR_DIRECTORY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
__TRAINPATH__ = os.path.join(PAR_DIRECTORY, "data", "processed", "smote_resampled_train_data.csv")
__TESTPATH__ = os.path.join(PAR_DIRECTORY, "data", "processed", "test_data.csv")

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

def save_model_locally(model, par_directory, run_name):
    """Save the model to a local directory under 'models'."""
    try:
        model_dir = os.path.join(par_directory, 'models')
        os.makedirs(model_dir, exist_ok=True)  # Create the directory if it doesn't exist
        model_path = os.path.join(model_dir, f"random_forest_{run_name}.pkl")

        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        logger.info(f"Model saved locally to {model_path}")
    except Exception as e:
        logger.error(f"Error saving model locally: {e}")
        raise

def train_and_log_model(X_train, y_train, X_test, y_test):
    """Train the model and log results with MLflow"""
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("run_name", run_name)

        trials = Trials()
        fmin_objective = lambda params: objective(params, X_train, y_train)
        best = fmin(fn=fmin_objective, space=SPACE, algo=tpe.suggest, max_evals=10, trials=trials)

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

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

        mlflow.log_params(best_params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

        signature = infer_signature(X_test, y_pred)
        mlflow.sklearn.log_model(model, "model", signature=signature)

        # Save the model locally
        save_model_locally(model, PAR_DIRECTORY, run_name)

        run_id = run.info.run_id
        logger.info(f"Run ID: {run_id}")
        return run_id

def main():
    """Main function to load data and run the training process"""
    X_train, y_train, X_test, y_test = load_data(__TRAINPATH__, __TESTPATH__)
    run_id = train_and_log_model(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()