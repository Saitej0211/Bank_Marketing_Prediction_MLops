import pickle
import pandas as pd
from google.cloud import storage
import os
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data", "processed")

KEY_PATH = os.path.join(PROJECT_DIR, "config", "key.json")
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = KEY_PATH

def load_preprocessing_objects(data_dir):
    """Load all preprocessing objects from local directory"""
    preprocessors = {}
    
    # Load label encoders for categorical columns
    categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 
                         'loan', 'contact', 'month']
    for col in categorical_columns:
        with open(f"{data_dir}\{col}_label_encoder.pkl", 'rb') as f:
            preprocessors[f'{col}_encoder'] = pickle.load(f)
    
    # Load scaler and normalizer
    with open(f"{data_dir}\scaler.pkl", 'rb') as f:
        preprocessors['scaler'] = pickle.load(f)
    
    with open(f"{data_dir}\\normalizer.pkl", 'rb') as f:
        preprocessors['normalizer'] = pickle.load(f)
    
    return preprocessors

def preprocess_input(input_data, preprocessors):
    """Preprocess a single row of input data"""
    # Create DataFrame with single row
    df = pd.DataFrame([input_data])
    
    # Process categorical columns one by one
    df['job'] = preprocessors['job_encoder'].transform([input_data['job']])
    df['marital'] = preprocessors['marital_encoder'].transform([input_data['marital']])
    df['education'] = preprocessors['education_encoder'].transform([input_data['education']])
    df['default'] = preprocessors['default_encoder'].transform([input_data['default']])
    df['housing'] = preprocessors['housing_encoder'].transform([input_data['housing']])
    df['loan'] = preprocessors['loan_encoder'].transform([input_data['loan']])
    df['contact'] = preprocessors['contact_encoder'].transform([input_data['contact']])
    df['month'] = preprocessors['month_encoder'].transform([input_data['month']])
    # Scale numerical features
    numerical_features = df[['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']]
    # Normalize the entire dataset
    normalized_data = preprocessors['normalizer'].transform(df)
    # Convert to DataFrame with correct column order
    columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 
              'housing', 'loan', 'contact', 'day', 'month', 'duration', 
              'campaign', 'pdays', 'previous']
    final_df = pd.DataFrame(normalized_data, columns=columns)
    return final_df


def load_model_from_gcp():
    """Load model from GCP bucket"""
    storage_client = storage.Client()
    bucket = storage_client.bucket("mlopsprojectdatabucketgrp6")
    blob = bucket.blob("models/best_random_forest_model/model.pkl")
    model_bytes = blob.download_as_bytes()
    return pickle.loads(model_bytes)

def predict_single_row():
    input_data = {
        'age': 24,
        'job': 'technician',
        'marital': 'single',
        'education': 'secondary',
        'default': 'no',
        'balance': -103,
        'housing': 'yes',
        'loan': 'yes',
        'contact': 'unknown',
        'day': 15,
        'month': 'may',
        'duration': 145,
        'campaign': 1,
        'pdays': -1,
        'previous': 0
    }
    
    # Load model first to get feature names
    model = load_model_from_gcp()
    
    # Load preprocessing objects and process data
    preprocessors = load_preprocessing_objects(DATA_DIR)
    processed_data = preprocess_input(input_data, preprocessors)
    # Make prediction
    prediction = model.predict(processed_data)
    
    return prediction[0]
# Use the function
if __name__ == "__main__":
    try:
        result = predict_single_row()
        print(f"Prediction: {result}")
    except Exception as e:
        print(f"Error during prediction: {str(e)}")