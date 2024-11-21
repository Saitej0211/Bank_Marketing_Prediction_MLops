from flask import Flask, request, render_template
from google.cloud import storage
import pickle
import numpy as np
import os

# Initialize Flask app with custom template folder
template_dir = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=template_dir)

# Set up GCP credentials
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEY_PATH = os.path.join(PROJECT_DIR, "config", "key.json")
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = KEY_PATH

# Initialize Google Cloud Storage client
storage_client = storage.Client()
bucket_name = "mlopsprojectdatabucketgrp6"
model_blob_name = "models/best_random_forest_model/model.pkl"

def load_artifacts():
    bucket = storage_client.bucket(bucket_name)
    
    # Load model
    model_blob = bucket.blob("models/best_random_forest_model/model.pkl")
    model = pickle.loads(model_blob.download_as_bytes())
    
    # Load preprocessing artifacts from registered_model_meta
    meta_blob = bucket.blob("models/best_random_forest_model/registered_model_meta")
    preprocessor = pickle.loads(meta_blob.download_as_bytes())
    
    return model, preprocessor

def preprocess_input(data, preprocessor):
    # Convert categorical variables
    categorical_features = ['job', 'marital', 'education', 'default', 'housing', 
                            'loan', 'contact', 'month']
    
    # Create feature array in correct order
    features = np.array([
        data['age'], data['job'], data['marital'], data['education'],
        data['default'], data['balance'], data['housing'], data['loan'],
        data['contact'], data['day'], data['month'], data['duration'],
        data['campaign'], data['pdays'], data['previous']
    ]).reshape(1, -1)
    
    # Apply preprocessing
    processed_features = preprocessor.transform(features)
    return processed_features

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        # Get form data
        input_data = {
            'age': float(request.form.get('age', 0)),
            'job': request.form.get('job', ''),
            'marital': request.form.get('marital', ''),
            'education': request.form.get('education', ''),
            'default': request.form.get('default', ''),
            'balance': float(request.form.get('balance', 0)),
            'housing': request.form.get('housing', ''),
            'loan': request.form.get('loan', ''),
            'contact': request.form.get('contact', ''),
            'day': int(request.form.get('day', 0)),
            'month': request.form.get('month', ''),
            'duration': int(request.form.get('duration', 0)),
            'campaign': int(request.form.get('campaign', 0)),
            'pdays': int(request.form.get('pdays', 0)),
            'previous': int(request.form.get('previous', 0))
        }
        
        try:
            # Load model and preprocessor
            model, preprocessor = load_artifacts()
            
            # Preprocess the input
            processed_input = preprocess_input(input_data, preprocessor)
            
            # Make prediction
            prediction = model.predict(processed_input)[0]
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            prediction = None

    return render_template('index.html', prediction=prediction)

@app.route('/health')
def health_check():
    return {"status": "ok"}, 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)