#!/bin/bash

# Update package lists
sudo apt-get update

# Install required packages
sudo apt-get install -y python3 python3-pip python3.11-venv git

# Create directory and set permissions
sudo mkdir -p /home/bank-marketing-prediction-mlops
sudo chmod -R 777 /home/bank-marketing-prediction-mlops

# Change to the project directory
cd /home/bank-marketing-prediction-mlops

# Clone the repository
git clone --single-branch --branch hashwanth_modeldeployment https://github.com/Saitej0211/Bank_Marketing_Prediction_MLops.git

# Navigate to the gcpdeploy directory
cd Bank_Marketing_Prediction_MLops/gcpdeploy

# Create virtual environment
python3 -m venv env

# Activate virtual environment
. env/bin/activate

# Install requirements (removed --user flag)
pip install -r requirements.txt

echo "Setup completed successfully"