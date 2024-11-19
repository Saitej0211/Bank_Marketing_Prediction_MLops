import os
import pickle
import pandas as pd
import json
import logging
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

# Logging Declarations
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

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

        logger.info(f"Loading X from {X_path}")
        logger.info(f"Loading y from {y_path}")
        
        # Load test data
        X_data = pd.read_csv(X_path)
        y_data = pd.read_csv(y_path).squeeze()  # Use squeeze to get a Series if y is single-column

        # Ensure that 'y' (target column) is not present in X
        if 'y' in X_data.columns:
            X_data = X_data.drop(columns=['y'])

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
        
        return importance_df
    
    except Exception as e:
        logger.exception(f"Error during sensitivity analysis: {e}")
        return None
    
if __name__ == "__main__":
    logger.info("Starting Sensitivity analysis")
    importance_df = perform_sensitivity_analysis()
    if importance_df is not None:
        logger.info(f"Sensitivity analysis results: {importance_df}")
    logger.info("Completed Sensitivity analysis")