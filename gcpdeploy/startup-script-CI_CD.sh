#!/bin/bash

# Log file paths
APP_LOG="/var/log/bank_marketing_app-CI_CD.log"
ERROR_LOG="/var/log/bank_marketing_app_error-CI_CD.log"

# Ensure necessary directories exist
mkdir -p /var/log

# Log script execution start
echo "Script execution started at $(date)" | tee -a $APP_LOG

# Run the setup script
chmod +x /home/bank-marketing-prediction-mlops/Bank_Marketing_Prediction_MLops/gcpdeploy/setup-CI_CD.sh
/home/bank-marketing-prediction-mlops/Bank_Marketing_Prediction_MLops/gcpdeploy/setup-CI_CD.sh >> $APP_LOG 2>> $ERROR_LOG
if [ $? -ne 0 ]; then
    echo "Setup script failed. Check error logs: $ERROR_LOG" | tee -a $ERROR_LOG
    exit 1
fi

# Navigate to the application directory
APP_DIR="/home/bank-marketing-prediction-mlops/Bank_Marketing_Prediction_MLops/gcpdeploy"
if [ -d "$APP_DIR" ]; then
    cd "$APP_DIR"
else
    echo "Application directory not found: $APP_DIR. Exiting." | tee -a $ERROR_LOG
    exit 1
fi

# Activate virtual environment
VENV_DIR="$APP_DIR/env"
if [ -d "$VENV_DIR" ]; then
    . "$VENV_DIR/bin/activate"
else
    echo "Virtual environment not found: $VENV_DIR. Exiting." | tee -a $ERROR_LOG
    exit 1
fi

# Run the application in the background
nohup python3 app.py >> $APP_LOG 2>> $ERROR_LOG &
APP_PID=$!

# Log success or failure
if ps -p $APP_PID > /dev/null; then
    echo "Application started successfully with PID $APP_PID. Logs: $APP_LOG" | tee -a $APP_LOG
else
    echo "Failed to start the application. Check error logs: $ERROR_LOG" | tee -a $ERROR_LOG
    exit 1
fi

# Log script execution end
echo "Script execution completed at $(date)" | tee -a $APP_LOG
