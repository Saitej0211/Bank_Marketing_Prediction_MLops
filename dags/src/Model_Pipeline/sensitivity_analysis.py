import os
import pickle
import pandas as pd
import json
import logging
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance

# Logging Declarations
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "bias_analysis_output")
os.makedirs(OUTPUT_DIR, exist_ok=True) # Create the directory if it doesn't exist

def perform_sensitivity_analysis(model_path, X_path, y_path):
    """
    Perform sensitivity analysis to understand how variations in features affect the model's predictions.
    
    Parameters:
    - model_path: The path to the trained model.
    - X_path: The feature matrix for the test data.
    - y_path: The target vector for the test data.
    """
    logger.info(f"Best Model Path: {model_path}")
    try:
        # Load the trained model
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
        logger.info(f"Loaded model from {model_path}")

        logger.info(f"Model:  {model}")

        # Extract hyperparameters from the model metadata
        hyperparams = {key: value for key, value in model.get_params().items()}
        #logger.info(f"Loaded hyperparameters: {json.dumps(hyperparams, indent=4)}")

        logger.info(f"Loading X from {X_path}")
        logger.info(f"Loading y from {y_path}")
        
        # Load test data
        X_data = pd.read_csv(X_path)
        y_data = pd.read_csv(y_path).squeeze()  # Use squeeze to get a Series if y is single-column

        # Ensure that 'y' (target column) is not present in X
        if 'y' in X_data.columns:
            X_data = X_data.drop(columns=['y'])

        logger.info(f"Model for Feature Sensitivity Analysis: {model}")
        # Perform permutation importance to evaluate feature importance
        result = permutation_importance(model, X_data, y_data, n_repeats=10, random_state=42, n_jobs=-1)
        
        # Store feature importances and their corresponding scores
        feature_importances = result.importances_mean
        std_importances = result.importances_std
        features = X_data.columns
        
        # Log feature importance values
        for feature, importance, std in zip(features, feature_importances, std_importances):
            logger.info(f"Feature: {feature}, Importance: {importance:.4f}, Standard Deviation: {std:.4f}")
        
        # Create a DataFrame for better visualization of results
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': feature_importances,
            'Std Dev': std_importances
        })

        # Sort features by importance
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        # Plot the feature importance
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['Feature'], importance_df['Importance'], xerr=importance_df['Std Dev'], color='skyblue')
        plt.xlabel('Importance')
        plt.title('Feature Importance (Permutation Importance)')
        plt.tight_layout()
        plt.show()

        # Save the feature sensitivity plot
        feature_plot_path = os.path.join(OUTPUT_DIR, "feature_sensitivity.png")
        plt.savefig(feature_plot_path)
        plt.close()
        logger.info(f"Feature sensitivity plot saved to {feature_plot_path}")

        logger.info(f"Model for Hyper-Parameter Sensitivity Analysis: {model}")
        
        # Define the parameters to analyze and their current values
        hyperparams_to_analyze = {
            'max_depth': model.get_params().get('max_depth'),
            'min_samples_leaf': model.get_params().get('min_samples_leaf'),
            'min_samples_split': model.get_params().get('min_samples_split'),
            'n_estimators': model.get_params().get('n_estimators'),
        }

        logger.info(f"Hyperparameters for Sensitivity Analysis: {hyperparams_to_analyze}")

        # Storage for sensitivity analysis results
        hyperparam_sensitivity = {}

        for param, original_value in hyperparams_to_analyze.items():
            logger.info(f"Analyzing parameter: {param} (original value: {original_value})")

            # Generate variations around the original value (±50%)
            variations = np.linspace(original_value * 0.5, original_value * 1.5, num=5).astype(int)
            param_impact = []

            for value in variations:
                try:
                    # Temporarily update the parameter
                    temp_model = model.set_params(**{param: value})
                    temp_model.fit(X_data, y_data)  # Fit with updated parameter
                    
                    # Evaluate the model (using accuracy as an example metric)
                    y_pred = temp_model.predict(X_data)
                    score = accuracy_score(y_data, y_pred)
                    param_impact.append((value, score))
                    
                    logger.info(f"Parameter: {param}, Value: {value}, Accuracy: {score:.4f}")
                except Exception as e:
                    logger.error(f"Error analyzing {param} with value {value}: {e}")
            
            hyperparam_sensitivity[param] = param_impact

        logger.info("Hyper-Parameter Sensitivity Analysis Complete")
        logger.info(f"hyperparam_sensitivity: {hyperparam_sensitivity}")
        # Log summary results
        for param, impacts in hyperparam_sensitivity.items():
            logger.info(f"Sensitivity Results for {param}: {impacts}")

        # Plot hyperparameter sensitivity
        for param, results in hyperparam_sensitivity.items():
            variations, impacts = zip(*results)  # Unpack results into variations and impacts
            
            if len(variations) != len(impacts):
                logger.error(f"Mismatch in dimensions for {param}: Variations {len(variations)}, Impacts {len(impacts)}")
                continue  # Skip this parameter to avoid the plotting error

            try:
                plt.plot(variations, impacts, label=param)
            except ValueError as e:
                logger.error(f"Error plotting {param}: {e}")

        plt.xlabel("Hyperparameter Value")
        plt.ylabel("Metric (Accuracy)")
        plt.title("Hyperparameter Sensitivity Analysis")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Save the hyperparameter sensitivity plot
        hyperparam_plot_path = os.path.join(OUTPUT_DIR, "hyperparameter_sensitivity.png")
        plt.savefig(hyperparam_plot_path)
        plt.close()
        logger.info(f"Hyperparameter sensitivity plot saved to {hyperparam_plot_path}")
        

        logger.info("Completed Sensitivity analysis")
        return importance_df, hyperparam_sensitivity
    
    except Exception as e:
        logger.exception(f"Error during sensitivity analysis: {e}")
        return None
    
if __name__ == "__main__":
    logger.info("Starting Sensitivity analysis")
    importance_df, hyperparam_sensitivity = perform_sensitivity_analysis()

    logger.info("Completed Sensitivity analysis")