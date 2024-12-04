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

# Clone the GitHub repository
if [ ! -d "Bank_Marketing_Prediction_MLops" ]; then
  git clone https://github.com/Saitej0211/Bank_Marketing_Prediction_MLops.git
else
  echo "Repository already exists. Pulling latest changes..."
  cd Bank_Marketing_Prediction_MLops
  git pull origin main
fi

# Navigate to the gcpdeploy directory
cd /home/bank-marketing-prediction-mlops/Bank_Marketing_Prediction_MLops/gcpdeploy

# Create virtual environment
if [ ! -d "env" ]; then
  python3 -m venv env
fi

# Activate virtual environment
. env/bin/activate

# Install requirements
pip install -r requirements.txt

# Copy the startup script to the VM's home directory
cp startup-script-CI_CD.sh /home/${USER}/startup-script-CI_CD.sh
chmod +x /home/${USER}/startup-script-CI_CD.sh

echo "Setup completed successfully"
