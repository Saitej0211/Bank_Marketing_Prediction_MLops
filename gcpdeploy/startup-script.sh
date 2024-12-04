#!/bin/bash

# Navigate to the project directory and gcpdeploy folder
cd /home/bank-marketing-prediction-mlops/Bank_Marketing_Prediction_MLops/gcpdeploy

# Activate the virtual environment
. env/bin/activate

# Run the application in the background using nohup
nohup python3 app.py & 