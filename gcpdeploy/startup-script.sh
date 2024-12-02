#!/bin/bash

# Log file paths
APP_LOG="/var/log/bank_marketing_app.log"
ERROR_LOG="/var/log/bank_marketing_app_error.log"

# Ensure necessary directories exist
mkdir -p /var/log

# Run the setup script
chmod +x /home/bank-marketing-prediction-mlops/setup.sh
/home/bank-marketing-prediction-mlops/setup.sh > $APP_LOG 2>> $ERROR_LOG || {
    echo "Setup script failed. Check error logs: $ERROR_LOG" | tee -a $ERROR_LOG
    exit 1
}

# Navigate to the application directory
cd /home/bank-marketing-prediction-mlops/Bank_Marketing_Prediction_MLops/gcpdeploy || {
    echo "Failed to navigate to application directory. Exiting." | tee -a $ERROR_LOG
    exit 1
}

# Activate virtual environment
if [ -d "env" ]; then
    . env/bin/activate
else
    echo "Virtual environment not found. Exiting." | tee -a $ERROR_LOG
    exit 1
}

# Run the application in the background
nohup python3 app.py > $APP_LOG 2>> $ERROR_LOG &

# Log success message
if [ $? -eq 0 ]; then
    echo "Application started successfully. Logs: $APP_LOG" | tee -a $APP_LOG
else
    echo "Failed to start the application. Check error logs: $ERROR_LOG" | tee -a $ERROR_LOG
    exit 1
}
