# Bank Marketing Campaign Prediction
Sheela Hansda, Saiteja Reddy Gajula, Hashwanth Moorthy, Aishwariya Alagesan, Vignesh Sankar, Tejesvani Muppara Vijayaram

# Introduction

## Business Problem:

This competitive financial industry needs attraction and retention of clients, and one key offering of banks comes in the shape of term deposit accounts. These accounts lead to steady streams of revenue while developing client loyalty. To drive subscriptions, the bank will carry out direct marketing campaigns through calls, e-mails, and other means of contact with the clients. However, not all customers are equal, and as such, a significant amount of the marketing effort could be focused on those customers who have relatively lower interest in such products. 

As a result, this usually culminates in resource wastage, poor use of time, and suboptimal return on marketing investment. The challenge, therefore, is to identify which customers are most likely to subscribe based on historical data that will help the bank fine-tune its targeting strategy. A more focused approach would limit unproductive contacts, reduce costs, and enhance customer experience by concentrating on those who are more open toward term deposits, thereby creating a win for the bank and the clients.

## AI Solution Using Binary Classification Model:

To handle such a problem, the AI-driven binary classification model may be implemented that will label each client as likely to subscribe-yes or not-no based on a set of varied features. This model leverages historical campaign data, including client demographics such as age, income, and job type, their past interactions with the bank, and also broader socio-economic indicators such as employment rate and market conditions at the time of contact. With this, the model learns patterns and correlations that no human marketer would find. 

Hence, the model has learned what triggers increase the likelihood of subscription. When trained, it will then give an output for new clients with high accuracy; at this point, targeting by the bank can be restricted to those most likely to respond. This approach allows the bank to effectively enhance campaign efficiency, prioritize resources, and higher success rates of term deposit subscriptions, all while reducing overall cost and resources spent on less promising leads. The model also offers actionable insights into client behavior, thus enabling further refinement of future marketing strategies and deeper understanding of the client base.

The primary goal of this project is to develop a predictive model that accurately determines whether a client will subscribe to a term deposit based on past marketing campaign data. By using a binary classification approach, the project aims to optimize the bank's marketing strategy, improve the efficiency of client outreach, and increase the conversion rate of campaigns. This solution should ultimately enable the bank to make data-driven decisions that reduce costs, enhance customer engagement, and maximize return on investment for its marketing efforts.

# Dataset Information
## Data Card

Shape - (45211, 16)

## Variable Information

| Variable Name  | Role     | Type        | Description                                                                                                     | Units           | Missing Values |
|----------------|----------|-------------|-----------------------------------------------------------------------------------------------------------------|-----------------|----------------|
| age            | Feature  | Integer     | Age of the client.                                                                                              | -               | No             |
| job            | Feature  | Categorical | Type of job (e.g., 'admin.','blue-collar','entrepreneur', etc.)                                                 | -               | No             |
| marital        | Feature  | Categorical | Marital status (e.g., 'divorced','married','single'; 'divorced' includes both divorced and widowed)             | -               | No             |
| education      | Feature  | Categorical | Education level (e.g., 'basic.4y','basic.6y','high.school','university.degree', etc.)                           | -               | No             |
| default        | Feature  | Binary      | Whether the client has credit in default (yes/no).                                                              | -               | No             |
| balance        | Feature  | Integer     | Average yearly balance of the client’s account.                                                                 | Euros           | No             |
| housing        | Feature  | Binary      | Whether the client has a housing loan (yes/no).                                                                 | -               | No             |
| loan           | Feature  | Binary      | Whether the client has a personal loan (yes/no).                                                                | -               | No             |
| contact        | Feature  | Categorical | Contact communication type (e.g., 'cellular', 'telephone').                                                     | -               | Yes            |
| day_of_week    | Feature  | Date        | Day of the week the client was last contacted.                                                                  | -               | No             |
| month          | Feature  | Date        | Month of the year the client was last contacted (e.g., 'jan', 'feb', etc.).                                     | -               | No             |
| duration       | Feature  | Integer     | Duration of the last contact in seconds. Note: Affects the output target but should be excluded in training for realistic predictions. | Seconds         | No             |
| campaign       | Feature  | Integer     | Number of contacts performed during the current campaign (including the last contact).                          | -               | No             |
| pdays          | Feature  | Integer     | Days since the client was last contacted in a previous campaign (-1 indicates no prior contact).                | -               | Yes            |
| previous       | Feature  | Integer     | Number of contacts performed before this campaign for this client.                                              | -               | No             |
| poutcome       | Feature  | Categorical | Outcome of the previous marketing campaign (e.g., 'failure', 'nonexistent', 'success').                         | -               | Yes            |
| y              | Target   | Binary      | Whether the client subscribed to a term deposit as a result of this campaign (yes/no).                          | -               | No             |

## Datasource Link: 
Source:  https://archive.ics.uci.edu/dataset/222/bank+marketing

## Project Structure
- **assets/**: Stores images, visualizations, and graphs.
- **data/**: Contains raw and processed datasets.
- **notebooks/**: Jupyter notebooks for exploratory data analysis (EDA) and model experimentation.
- **dags/**: Source code for data preprocessing, feature engineering, and Airflow DAG.
- **config/**: Configuration files for setting paths, parameters, and schemas.
- **tests/**: Unit tests for validating code functionality.
- **logs/**: Stores log files from pipeline runs.


## Git repository structure
```plaintext
.
├── LICENSE                         ## License information
├── README.md                       ## Overview of the project, how to use it, and dataset details
├── .github\workflows               ## Contains file for workflows
│   ├── CI_CD_gcp_deploy.yml        ## Workflow to automate GCP deployment using Github actions
├── assets                          ## Store images, graphs, or other visualizations
│   ├── images                      ## Images used in the project
│   └── plots                       ## Graphs or plots generated during analysis
├── config                          ## Configuration files for the project
│   ├── config.yaml                 ## General configurations like file paths, hyperparameters
│   ├── constants.py                ## Constant variables (e.g., target variable, feature names)
│   ├── Key.json                    ## Key file (possibly for authentication or API access)
│   └── schema.yaml                 ## Dataset schema (columns and types)
├── data                            ## Raw and processed data files
│   ├── raw                         ## Raw dataset files
│   │   ├── bank                    ## Raw bank marketing data
│   │   └── bank-additional          ## Additional raw bank marketing data
│   ├── processed                   ## Processed data, train/test splits
│       ├── eda_plots               ## Exploratory Data Analysis plots
│       ├── correlation_matrix.csv  ## Correlation matrix in CSV format
│       ├── correlation_matrix.pkl  ## Correlation matrix in pickle format
│       ├── dataframe_description.csv  ## Description of the dataframe
│       ├── dataframe_info.csv      ## Information about the dataframe
│       ├── datatype_format_processed.csv  ## Datatype format after processing (CSV)
│       ├── datatype_format_processed.pkl  ## Datatype format after processing (Pickle)
│       ├── datatype_info_after.csv  ## Datatype information after processing (CSV)
│       ├── datatype_info_before.csv  ## Datatype information before processing (CSV)
│       ├── encoded_data.csv        ## Encoded data in CSV format
│       ├── encoded_data.pkl        ## Encoded data in pickle format
│       ├── outlier_handled_data.csv  ## Data after handling outliers (CSV)
│       ├── outlier_handled_data.pkl  ## Data after handling outliers (Pickle)
│       ├── processed_data.pkl      ## Final processed data in pickle format
│       ├── raw_data.csv            ## Raw dataset in CSV format
│       └── raw_data.pkl            ## Raw dataset in pickle format  
├── dags
│   ├── logs                        # Directory to store log files generated by Airflow during DAG execution.
│   │   ├── dag_id=DataPipeline         ## Logs related to DataPipeline DAG execution 
│   │   │   ├── dag_processor_manager       ## Logs related to DAG processor manager 
│   │   │   ├── download_data.log           ## Log for downloading data 
│   │   │   ├── eda.log                     ## Log for Exploratory Data Analysis 
│   │   │   └── outlier_handling.log        ## Log for outlier handling process 
│   │
│   ├── src                         # Main source directory containing all Python scripts for data preprocessing and │validation.
│   │   ├── __pycache__             # Directory for Python cache files automatically generated after running scripts.
│   │   │   └── # These files are generated by Python to speed up execution. No need to modify them manually.
│   │   │
│   │   ├── data_preprocessing      # Contains scripts for various data preprocessing tasks.
│   │   │   ├── __pycache__         # Cache files related to the data preprocessing module.
│   │   │   ├── __init__.py          # Initialization file for the `data_preprocessing` module.
│   │   │   ├── correlation_analysis.py  # Script to perform correlation analysis on features in the dataset.
│   │   │   │                           # Helps identify multicollinearity or redundant features.
│   │   │   ├── datatype_format.py       # Script to ensure correct data types for all columns in the dataset.
│   │   │   │                           # Handles type conversion and ensures consistency across datasets.
│   │   │   ├── encoding.py              # Script responsible for encoding categorical variables into numerical values.
│   │   │   │                           # Uses techniques like one-hot encoding or label encoding.
│   │   │   ├── outlier_handling.py      # Script to detect and handle outliers using statistical methods like z-score │or IQR.
│   │   │   ├── preprocessing_main.py    # Main script coordinating all preprocessing steps, such as handling missing │values,
│   │   │                                # encoding, and normalization of features.
│   │   │   ├── removing_duplicates.py   # Script to remove duplicate entries from the dataset to ensure data integrity.
│   │   │   ├── smote.py                 # Implements SMOTE (Synthetic Minority Over-sampling Technique) to address │class imbalance.
│   │
│   │   ├── Data_validation          # Contains scripts for validating the quality of input data before further │processing.
│   │       ├── __pycache__         # Cache files related to the data validation module.
│   │       ├── anomaly_detection.py     # Script to detect anomalies in the dataset using predefined rules or │statistical methods.
│   │       ├── data_schema_statistics_generation.py  # Script that generates statistics about dataset schema (e.g., │column types, missing values).
│   │       │                                        # Ensures incoming datasets adhere to expected formats.
│   │       ├── util_bank_marketing.py  # Utility functions specific to a bank marketing dataset, including custom │transformations or validations.
│   │   ├── Model_Pipeline          # Contains scripts for Model Development, Evaluation and Pushing to GCP registry
│   │       ├── __pycache__         # Cache files related to the data validation module.
│   │       ├── model_development_and_evaluation_with_mlflow.py     # Script to develop the model, tune hyperparameters and version the model with mlflow.
│   │       ├── compare_best_models.py  # Script that is used to compare the recently run model with the latest versioned model from mlflow and push the model to GCP registry.
│   │       ├── Bias check.py         #
|   |       ├── Sensitivity check.py   #                              
│   │       ├── push_to_gcp.py  # Utility functions specific to a bank marketing dataset, including custom │transformations or validations.
│   ├── data_pipeline.py                   # Main Airflow DAG definition file that orchestrates different tasks such as │downloading data,
│   │                                # preprocessing it, validating it, and loading it into a target system. Defines │task dependencies,schedules, and triggers within Airflow.
│   ├── model_development_pipeline.py      # Main Airflow DAG definition file that orchestrates different model development tasks such as │Model development, training, bias and sensitivity check, compare and push best model to GCP bucket. 
│   ├── DownloadData.py              # Script responsible for downloading raw datasets from external sources like APIs │or cloud storage systems.Ensures datasets are available locally before starting any preprocessing tasks.
│   ├── eda.py                       # Script dedicated to performing Exploratory Data Analysis (EDA) on raw datasets. │Generates summary statistics,
│   │                                # visualizations, and other insights needed before applying transformations.
│   ├── HandlingNullValues.py        # Script specifically designed to handle missing values in datasets. Applies │techniques like mean/mode imputation,
│   │                                 # forward/backward filling, or other imputation methods based on business logic.
│   ├── LoadData.py                  # After preprocessing and validation steps are complete, this script loads cleaned │datasets into a target system,such as a database or cloud storage service. The cleaned data can then be used for model │training or reporting.
│   │
│   ├── util_bank_marketing.py        # Utility script containing helper functions specific to handling bank marketing │datasets. These functions may include custom transformations, feature engineering, or domain-specific validations.
│   └── __init__.py                  # Initialization file for the overall `dags` module. This file allows other modules │within `dags` to be imported easily.
├──gcpdeploy
|   ├── _pycache_/                      ## Python cache files
|   ├── myenv/                          ## Virtual environment directory
|   ├── templates/                      ## Template files for the application
|   ├── app_test.py                     ## Application test script
|   ├── app.log                         ## Application log file
|   ├── app.py                          ## Main application file
|   ├── check.py                        ## Validation and health check script
|   ├── credentials.txt                 ## GCP authentication credentials
|   ├── Dockerfile                      ## Container configuration for VM updates
|   ├── load_test.py                    ## Performance testing script
|   ├── requirements.txt                ## Python package dependencies
|   ├── sampleCheck.py                  ## Sample validation script
|   ├── samplecheck1.ipynb              ## Jupyter notebook for testing
|   ├── setup-CI_CD.sh                  ## VM environment setup script for CI CD
|   ├── setup.sh                        ## VM environment setup script
|   ├── startup-script-CI_CD.sh         ## VM instance startup configuration for CI CD
|   ├── startup-script.sh               ## VM instance startup configuration
|   └── test.sh                         ## Test execution script
├── plugins                         ## Plugins used in the project (if any)
├── src                             ## Source code for the project
│   ├── models                      ## Machine learning models and related scripts 
│   │   └── __init__.py             ## Initialization file for models module 
│   └── utils                       ## Utility functions used throughout the project 
│       └── __init__.py             ## Initialization file for utils module 
├── tests                           ## Unit tests for various modules 
│   ├── test_CorrelationAndEncoding.py  ## Test script for correlation and encoding 
│   ├── test_data_format.py         ## Test script for data formatting 
│   ├── test_DownloadAndLoadData.py  ## Test script for downloading and loading data 
│   ├── test_eda.py                 ## Test script for exploratory data analysis 
│   ├── test_HandleNullValues.py    ## Test script for handling null values 
│   └── test_Smote.py               ## Test script for SMOTE oversampling technique 
├── .dvcignore                      ## DVC ignore file to specify which files to exclude from DVC tracking 
├── .env                            ## Environment variables file 
├── .gitattributes                  ## Git attributes configuration file 
├── .gitignore                      ## Git ignore file to exclude specific files from version control 
├── airflow.cfg                     ## Airflow configuration file  
├── docker-compose.yaml             ## Docker Compose setup for multi-container environments 
├── Dockerfile                      ## Docker container setup  
└── requirements.txt                ## List of dependencies (Python libraries)  
```

## Installation

This project requires Python >= 3.8. Please make sure that you have the correct Python version installed on your device. Additionally, this project is compatible with Windows, Linux, and Mac operating systems.

### Prerequisites

- git
- python>=3.8
- docker daemon/desktop should be running

### User Installation

The User Installation Steps are as follows:

1. Clone the git repository onto your local machine:
  ```
  git clone https://github.com/Saitej0211/Bank_Marketing_Prediction_MLops.git
  ```
2. Check if python version >= 3.8 using this command:
  ```
  python --version
  ```
3. Check if you have enough memory
  ```docker
  docker run --rm "debian:bullseye-slim" bash -c 'numfmt --to iec $(echo $(($(getconf _PHYS_PAGES) * $(getconf PAGE_SIZE))))'
  ```
**If you get the following error, please increase the allocation memory for docker.**
  ```
  Error: Task exited with return code -9 or zombie job
  ```
4. After cloning the git onto your local directory, please edit the `docker-compose.yaml` with the following changes:

  ```yaml
  user: "1000:0" # This is already present in the yaml file but if you get any error regarding the denied permissions feel free to edit this according to your uid and gid
  AIRFLOW__SMTP__SMTP_HOST: smtp.gmail.com # If you are using other than gmail to send/receive alerts change this according to the email provider.
  AIRFLOW__SMTP__SMTP_USER: # Enter your email 'don't put in quotes'
  AIRFLOW__SMTP__SMTP_PASSWORD: # Enter your password here generated from google in app password
  AIRFLOW__SMTP__SMTP_MAIL_FROM:  # Enter your email
 - ${AIRFLOW_PROJ_DIR:-.}/dags: #locate your dags folder path here (eg:/home/vineshgvk/PII-Data-1/dags)
 - ${AIRFLOW_PROJ_DIR:-.}/logs: #locate your project working directory folder path here (eg:/home/vineshgvk/PII-Data-1/logs)
 - ${AIRFLOW_PROJ_DIR:-.}/config: #locate the config file from airflow (eg:/home/vineshgvk/airflow/config)
  ```
5. In the cloned directory, navigate to the config directory under Bank_Marketing_Prediction_Mlops and place your key.json file from the GCP service account for handling pulling the data from GCP.

6. Build the Docker Image
   ```
   docker compose build
   ```
7. Run the Docker composer and initialize airflow.
   ```
   docker compose up ariflow-init
   ```
8. Run the docker image.
   ```
   docker compose up
   ```
9. To view Airflow dags on the web server, visit https://localhost:8080 and log in with credentials
   ```
   user: airflow2
   password: airflow2
   ```
10. Run the DAG by clicking on the play button on the right side of the window

11. Stop docker containers (hit Ctrl + C in the terminal)
    
# Tools Used for MLOps

- GitHub Actions
- Docker
- Airflow
- Data Version Control (DVC)
- Google Cloud Storage (GCS)

## GitHub Actions

GitHub Actions is configured to initiate workflows upon pushes and pull requests to any branch, including the "Name**" and main branches.

When a new commit is pushed, the workflow triggers a build process `unittest`. This process produces test reports in XML format, which are then stored as artifacts. The workflow is designed to locate and execute test cases situated within the test directory that correspond to modules in the dags directory. Additionally, the workflow assesses the code for readability, potential security issues, and adequate documentation. Upon the successful completion of these build checks , feature branches are merged into the main branch.

## Docker and Airflow

The `docker-compose.yaml` file contains the code neccessary to run Airflow. Through the use of Docker and containerization, we are able to ship our datapipeline with the required dependencies installed. This makes it platform indepedent, whether it is windows, mac or linux, our data pipeline should run smooth.

## Data Version Control (DVC)

DVC (Data Version Control) is an open-source tool essential for data versioning in machine learning projects. It tracks changes in datasets over time, ensuring reproducibility and traceability of experiments. By storing meta-information separately from data, DVC keeps Git repositories clean and lightweight. It integrates seamlessly with Git, allowing for efficient management of code, data, and models. This dual-repository approach simplifies collaboration and ensures that project states can be recreated easily. DVC's focus on data versioning is critical for maintaining the integrity and reliability of machine learning workflows.

## Google Cloud Platform (GCP)
We utilize Google Cloud Storage exclusively for storing our machine learning models, ensuring they are securely archived and readily accessible for deployment

One must set up a service account to use Google Cloud Platform services using below steps.

Go to the GCP Console: Visit the Google Cloud Console at https://console.cloud.google.com/.

Navigate to IAM & Admin > Service accounts: In the left-hand menu, click on "IAM & Admin" and then select "Service accounts".

Create a service account: Click on the "Create Service Account" button and follow the prompts. Give your service account a name and description.

Assign permissions: Assign the necessary permissions to your service account based on your requirements. You can either grant predefined roles or create custom roles.

Generate a key: After creating the service account, click on it from the list of service accounts. Then, navigate to the "Keys" tab. Click on the "Add key" dropdown and select "Create new key". Choose the key type (JSON is recommended) and click "Create". This will download the key file to your local machine.

[Link to our GCP bucket - It is Public](https://console.cloud.google.com/storage/browser/mlopsprojectdatabucketgrp6)

You can avoid these steps of creating a GCP bucket, instead you could raise a request to access our GCP bucket

![image](![image](https://github.com/user-attachments/assets/c2b3deb7-dba4-49ff-83ea-40be938e10e1)

Pictured above: GCP Bucket tracking the Data Version for the dataset

# End-to-End Pipeline for Model Deployment
![ML Project Pipeline](https://github.com/user-attachments/assets/84b79898-c4cb-41bd-a07c-60d07986216d)

Pictured above: ML project Pipeline

## Pipeline Optimization
![image](https://github.com/user-attachments/assets/f669d1e1-7d8e-422b-b6ea-d5d5ceb0af1a)
Pictured above: Airflow DAG Execution Gantt Chart for Data Pipeline. It is a popular project management tool used to visualize and track the progress of tasks or activities over time. It provides a graphical representation of a pipeline's schedule, showing when each task is planned to start and finish.

![image](https://github.com/user-attachments/assets/6f150050-db98-4011-9311-85cc989a9700)
Pictured above: Airflow DAG Execution Gantt Chart for Model Pipeline. 

Our Pipeline till this part for creating Data Pipeline has Data Collection ,Processing, Model Development, Logging, Versioning and Storing the Model to GCP
#### 1.Data Download
#### 2.Data Cleaning and Preprocessing

Our Pipeline begins with data acquisition where we fetch Data from the Source and perform Data Slicing. After the Data is downloaded, in the next step, we clean and Preprocess the Data. 
Next, the preprocessed data is analyzed to generate performance metrics.

We use Apache Airflow to orchestrate our data pipeline, treating each module as a distinct task within our primary DAG (Directed Acyclic Graph). This setup allows us to efficiently manage the flow from data acquisition to model deployment, ensuring each step is executed in the correct order and monitored for performance and success.

# Data Download
In this phase, the dataset is fetched and extracted into the designated data folder using the following modules:

- **DownloadData.py**: This python file downloads the latest available file from the GCP bucket and pickles it stores it in the folder.
- **LoadData.py**: This python file unpickles the data and stores it as a CSV file in the data/processed folder allowing the next task to pick this up as input for data validation.

# Statistics Generation
- **data_schema_statistics_generation.py**: Loads data from CSV file into a DataFrame and splits the data into training(60%), evaluation(20%), and serving(20%) datasets. Creates descriptive statistics for the training and serving datasets. Infers a training schema and serving schema(excluding the target variable Y) from the training statistics and saves it in .pbtxt format. Saves the data in .pkl format to be used for anomaly detection.

# Anomaly detection and alerts
- **anomaly_detection.py**: The `anomaly_detection` script detects anomalies in a dataset based on predefined rules for data quality and structure, logging any issues found. It loads data from a specified path, validates it against expected columns, data types, and value ranges, and saves results in .pkl and .csv formats. If anomalies are detected, the script sends an alert email with details of the issues. This script can be used as part of an automated Airflow DAG to streamline data validation workflows.
- **send_alerts_if_anomalies**: The system includes an email alert feature, which notifies the team immediately when an anomaly is detected. The alerts contain specific details about the detected issues, helping the team take quick corrective action.

![image](https://github.com/user-attachments/assets/5cae44b9-f35f-4d42-9ad0-14a7a057bf0c)

Pictured above : An alert message is sent to the recipient's email in case anomalies are detected.

# Handling Null values and duplicates

- **The process_data function performs the following operations**:
  
- **Data Loading**: Reads a CSV file into a pandas DataFrame.
- **Metadata Saving**: Saves DataFrame info and description to separate CSV files.
- **Duplicate Handling**:
Checks for duplicate rows in the DataFrame.
If duplicates are found, it removes them using drop_duplicates() method.
Logs the number of duplicate rows removed or if no duplicates were found.

- **Null and Unknown Value Handling**:
Identifies columns with high percentages of null or unknown values.
Drops columns with more than 80% null or unknown values.
Fills unknown values in 'job' and 'education' columns with mode values.
Data Saving: Saves the processed DataFrame as a pickle file.

# Data Cleaning and Preprocessing
Reliable and high-quality data is crucial for machine learning model performance. This preprocessing pipeline uses several scripts to ensure the data integrity and readiness for model training, making the pipeline more efficient and robust.

- **datatype_format.py**: Ensures data types are consistent across the dataset, improving compatibility with various analysis and machine learning techniques. It loads the processed data, checks and converts data types, logs data type changes, and saves the results.

- **outlier_handling.py**: Detects and handles outliers using the IQR method to reduce extreme values that could distort the model. It loads the processed data, calculates IQR for each numeric column, logs detected outliers, caps outliers within bounds, and saves the outlier-handled data as a .pkl file.

- **Encoding.py**: Encodes categorical variables to numeric formats for compatibility with machine learning algorithms. It loads the input data, identifies categorical columns, applies encoding transformations, and saves the processed data as .pkl and .csv files.

- **CorrelationAnalysis.py**: Analyzes and visualizes feature correlations to uncover relationships between variables. Loads the encoded data, computes the correlation matrix, saves it as .csv and .pkl files, and generates a heatmap visualization saved as an image.

- **Smote.py**: Addresses class imbalance using SMOTE to enhance model performance. Loads the encoded data, splits it into training and testing sets, applies SMOTE to balance classes, and saves the adjusted data.

# Email Alerts
- **Email Notification Setup**
The DAG is configured to send email notifications upon successful completion, allowing users to monitor task completion status directly from their inbox. This setup ensures users stay informed about the DAG's progress without manually checking the Airflow dashboard. To enable this feature, ensure that SMTP configurations in `docker-compose.yaml` file under the [smtp] section, specifying the SMTP server, email sender, and recipient settings.
```yaml
    AIRFLOW__SMTP__SMTP_HOST: smtp.gmail.com
    AIRFLOW__SMTP__SMTP_STARTTLS: 'True'
    AIRFLOW__SMTP__SMTP_SSL: 'False'
    AIRFLOW__SMTP__SMTP_USER: aishwariya.alagesanus@gmail.com           # Replace with your Gmail
    AIRFLOW__SMTP__SMTP_PASSWORD: **** **** **** ****         # App password generated from Google
    AIRFLOW__SMTP__SMTP_PORT: 587
    AIRFLOW__SMTP__SMTP_MAIL_FROM: aishwariya.alagesanus@gmail.com      # Same Gmail address
    AIRFLOW__SMTP__SMTP_TIMEOUT: 30                         # SMTP timeout in seconds
    AIRFLOW__SMTP__SMTP_RETRY_LIMIT: 5                      # Maximum retries
    AIRFLOW__SCHEDULER__ENABLE_HEALTH_CHECK: 'true'
  ```
- **Testing and Verification**
Once the DAG is executed, check for a "DAG Completed" email notification in the specified email inbox. This email serves as a confirmation that the DAG has successfully completed all tasks as per the defined workflow. Users can modify the notification logic to send emails upon task failures or retries, enhancing monitoring capabilities for critical workflows.

![image](https://github.com/user-attachments/assets/4435684c-a1e0-4fe2-8f0c-21ec5b14994c)

Pictured above : Dag completed notification sent to recipient mail.

# Model Building Evaluation and Retraining:

- **model_development_and_evaluation_with_mlflow.py**: This script performs hyperparameter tuning for a Random Forest model using Hyperopt. It tracks and logs the performance of different hyperparameter configurations (e.g., n_estimators, max_depth, min_samples_split) in MLflow. The results, including metrics like accuracy, precision, recall, and F1 score, are visualized in MLflow for comparison across multiple runs.

- **model_pipeline.py**: This DAG orchestrates the model development process, starting with training the model using run_model_development and logging metrics. If the performance metrics (e.g., accuracy, precision, recall, F1 score) fall below 0.7, the model retrains up to three times.

- **model_bias_detection.py**: This script performs bias analysis on a trained machine learning model by evaluating its performance across slices of sensitive features (e.g., age, marital status). It loads test data, detects bias by calculating metrics like accuracy and F1 score for each slice, and visualizes the results as bar plots. Outputs are saved in the bias_analysis_output directory for further review.

- **Sensitivity_analysis.py**: This script performs sensitivity analysis to assess how variations in input features impact the predictions of a trained machine learning model. It leverages permutation importance to rank feature contributions, highlighting the significance and variability of each feature in the model’s decision-making process. The results are logged and visualized as bar plots for clear interpretation.

- **compare_best_models.py**: This script compares multiple Random Forest models based on their metrics (e.g., accuracy) and selects the best-performing model. The selected model is logged, registered, and transitioned to the “Staging” stage in MLflow, ensuring streamlined model management and deployment.

- **push_to_gcp.py**: This script automates downloading the latest version of a machine learning model from the MLflow Model Registry and uploading it to Google Cloud Storage (GCS). It ensures seamless integration with MLflow and GCS, while logging each step for transparency and debugging.

# Hyper Parameter Tuning
This model has several hyperparameters, including max_depth, max_features, min_samples_split, min_samples_leaf, and n_estimators. These hyper parameters are tuning the hyperopts parameter in Sckit Learn and to choose the best model. We used MLflow to track different training runs by logging these hyperparameters and performance metrics such as accuracy, F1 score, precision, and recall.

# Staging, Production and Archived models (MLFLOW)
For monitoring our experimental machine learning pipeline, we employ MLflow, Docker, and Python. We selected three key metrics to determine the optimal model parameters from the plot provided:

![image](https://github.com/user-attachments/assets/06dc6ff6-6cc2-4f01-8c57-c58f7d2e9e5a)

Pictured: Plot for Performance Metrics Visualization of the Model

# Model Analysis
## Model Sensitivity Analysis

This project implements sensitivity analysis techniques to understand how different features and hyperparameters affect our machine learning model's performance. The analysis helps identify which inputs have the most significant impact on the model's predictions and how changes in hyperparameters influence overall model performance.

## Features

1. Feature Importance Analysis
   - Uses permutation importance to evaluate the impact of each feature on model performance
   - Visualizes feature importance using bar plots
     ![image](https://github.com/user-attachments/assets/e794b551-081c-4e9f-985f-08a3c11d285c)

2. Hyperparameter Sensitivity Analysis
   - Evaluates the effect of varying key hyperparameters (e.g., max_depth, min_samples_leaf) on the model's performance.
   - Visualizes the impact of each hyperparameter on model performance
     ![image](https://github.com/user-attachments/assets/d79852d7-5a37-4f2a-84a7-a4711090ac14)


## Model Bias Detection (Using Slicing Techniques)

Our project implements a comprehensive bias detection process to ensure fairness across different subgroups of the population. Here's an overview of our approach:

### 1. Performing Slicing

We break down the dataset into meaningful slices based on sensitive features, primarily focusing on age and marital status:

- Age groups: 18-30, 31-45, 46-60, 61-75, 75+
- Marital status: Single, Married, Divorced

The `slice_data` function handles this slicing process, creating appropriate bins for numerical features and using unique values for categorical features.

### 2. Tracking Metrics Across Slices

For each slice, we evaluate and track the following metrics:

- Accuracy
- Precision
- Recall
- F1-score

The `evaluate_slice` function calculates these metrics for each slice, allowing us to identify any significant disparities across different subgroups.

### 3. Visualizing Bias

We generate visualizations to illustrate performance across different slices:

- Bar plots are created for each sensitive feature, showing the performance metrics across slices.
- Visualizations are saved in the `bias_analysis_output` directory.

![bias_age_accuracy](https://github.com/user-attachments/assets/08f310ff-c206-4c8b-a872-7514bbd5b6f1)
![bias_marital_accuracy](https://github.com/user-attachments/assets/36f61390-15f8-4993-8373-14a829a87ee0)

### 4. Documenting Bias Detection

Our bias detection process is thoroughly documented:

- Detailed logging is implemented throughout the process, capturing information about each step.
- The `run_bias_analysis` function provides a comprehensive analysis of bias across specified sensitive features.

# Model Deployment

### **Step 1: Create a VM Instance and Set Up Environment**

1. **Create a VM instance:**
    - Go to the Google Cloud Console: [https://console.cloud.google.com/](https://console.cloud.google.com/).
    - Navigate to Compute Engine > VM instances.
    - Click "Create Instance".
    - Configure the instance:
        - **Name:** `bank-marketing-prediction-mlops-vm-instance`
        - **Region:** `us-central1`
        - **Zone:** `us-central1-a` (or any other compatible zone in the `us-central1` region)
        - **Machine type:** `e2-micro` (1 vCPU, 1 GB memory)
        - **Boot disk:** Debian GNU/Linux 10 (buster)
        - **Boot disk size:** 10 GB
    - Click "Create".

2. **Set up the environment:**
    - SSH into the VM instance.
    - Execute the environment setup script `gcpdeploy/setup.sh` to install necessary packages and set up the virtual environment:
    
    ```sh
    #!/bin/bash
    sudo apt-get update
    sudo apt-get install -y python3 python3-pip python3.11-venv git
    sudo mkdir -p /home/bank-marketing-prediction-mlops
    sudo chmod -R 777 /home/bank-marketing-prediction-mlops
    cd /home/bank-marketing-prediction-mlops
    git clone https://github.com/Saitej0211/Bank_Marketing_Prediction_MLops.git
    cd Bank_Marketing_Prediction_MLops/gcpdeploy
    python3 -m venv env
    . env/bin/activate
    pip install -r requirements.txt
    echo "Setup completed successfully"
    ```
### **Step 2: Create and Use a VM Snapshot**

1. **Create a snapshot of the VM instance:**
    - Stop the VM instance.
    - Go to Compute Engine > Snapshots.
    - Click "Create Snapshot".
    - Configure the snapshot:
        - **Name:** `bank-marketing-prediction-mlops-vm-snapshot`
        - **Source disk:** `bank-marketing-prediction-mlops-vm-instance` boot disk
    - Click "Create".

### **Step 3: Create a Custom VM Image**

1. **Create a custom image from the VM instance:**
    - Deactivate the virtual environment:
    
    ```sh
    deactivate
    ```
    - Stop the VM instance.
    - Go to Compute Engine > Images.
    - Click "Create Image".
    - Configure the image:
        - **Name:** `bank-marketing-prediction-image`
        - **Source:** `Disk` > `bank-marketing-prediction-mlops-vm-instance` boot disk
    - Click "Create"

### **Step 4: Configure Networking and Security**

1. **Set up VPC and subnets:**
    - Navigate to VPC Network > VPC Networks.
    - Click "Create VPC Network".
    - Configure the VPC:
        - **Name:** `bank-marketing-prediction-vpc`
        - **Subnets:** Custom
        - Click "Add subnet" to add a new subnet:
            - **Name:** `bank-marketing-prediction-vpc-subnet`
            - **Region:** `us-central1`
            - **IP address range:** `10.0.0.0/24`
            - **Purpose:** Private
        - Click "Done".
    - Click "Create".

2. **Set up firewall rules:**
    - Navigate to VPC Network > Firewall Rules.
    - Click "Create Firewall Rule".
    - Configure the rule:
        - **Name:** `bank-marketing-prediction-vpc-allow-custom`
        - **Network:** `bank-marketing-prediction-vpc`
        - **Direction of traffic:** Ingress
        - **Action on match:** Allow
        - **Targets:** All instances in the network
        - **Source IP ranges:** `0.0.0.0/0`
        - **Protocols and ports:**
            - Specified protocols and ports:
                - `tcp:22` (SSH)
                - `tcp:80` (HTTP)
                - `tcp:8000` (for FastAPI)
    - Click "Create".

### **Step 5: Create an Instance Template and Managed Instance Group (MIG)**

1. **Create a startup script:**
   Create a shell script that activates the virtual environment and runs the FastAPI server. Save this script as `startup-script.sh`:

    ```sh
    #!/bin/bash
    # Navigate to the project directory and gcpdeploy folder
    cd /home/bank-marketing-prediction-mlops/Bank_Marketing_Prediction_MLops/gcpdeploy
    # Activate the virtual environment
    . env/bin/activate
    # Run the application in the background using nohup
    nohup python3 app.py & 
    ```
2. **Create an instance template:**
    - Navigate to Compute Engine > Instance Templates.
    - Click "Create Instance Template".
    - Configure the template:
        - **Name:** `bank-marketing-prediction-template`
        - **Machine type:** `e2-micro` (1 vCPU, 1 GB memory)
        - **Boot disk:** Custom image `bank-marketing-prediction-image`
        - **Management, security, disks, networking, sole tenancy:**
            - Click "Networking"
            - In the "Networking" tab, select the VPC network `bank-marketing-prediction-vpc` and subnet `bank-marketing-prediction-vpc-subnet` you created earlier
            - Ensure to remove the pre-existing default vpc 

    - In the "Management, security, disks, networking, sole tenancy" section, find the "Automation" tab.
    - In the "Startup script" section, paste the contents of `startup-script.sh`.
    - Click "Create".

3. **Create a managed instance group (MIG):**
    - Go to Compute Engine > Instance Groups.
    - Click "Create Instance Group".
    - Configure the group:
        - **Name:** `bank-marketing-mig`
        - **Location:** Single zone
        - **Zone:** `us-central1-c` (or the same zone used for the custom VM image)
        - **Instance template:** `bank-marketing-prediction-template`
        - **Autoscaling policy:**
            - Target CPU utilization: 60%
            - Minimum number of instances: 1
            - Maximum number of instances: 3
    - Click "Create".

### **Step 6: Configure Load Balancer**
1. **Set up a load balancer:**
    - Navigate to **Network Services** > **Load balancing**.
    - Click "Create Load Balancer".
    - Select `HTTP(S) Load Balancing`.
    - Click `Start configuration`.
    - Choose `Global`.

2. **Configure the backend service:**
    - Select `Backend services & backend buckets`.
    - Click `Create a backend service`.
    - Configure the backend service:
        - **Name:** `bank-marketing-backend-service`
        - **Backend type:** `Instance group`
        - **Instance group:** `bank-marketing-mig`
        - **Port numbers:** `8000`
    - Configure the health check:
        - **Protocol:** `HTTP`
        - **Port:** `8000`
        - **Request path:** `/health` (should respond with status: ok)
    - Click `Create`.

3. **Configure the frontend:**
    - Click `Frontends`.
    - Click `Create a frontend IP and port`.
    - Configure the frontend:
        - **Name:** `bank-marketing-frontend`
        - **Protocol:** `HTTP`
        - **IP version:** `IPv4`
        - **Port:** `80`
    - Click `Done`.
  
4. **Finalize and create the load balancer:**
    - Review the configuration.
    - Click `Create`.

### **Step 7: Auto-scaling and Load Testing**
Auto-scaling automatically adjusts the number of VM instances in a group based on load or other metrics, ensuring that your application can handle varying amounts of traffic. Load testing simulates user traffic to ensure the system can handle real-world usage.

1. **Verify the setup:**
    - The startup script in the MIG will automatically activate the virtual environment and start the FastAPI server.
    - You can SSH into a VM instance from the MIG to check the logs and verify that the server is running.
    - Check the log file `/tmp/startup.log` for any errors.

2. **Test the load balancer:**
    - Send HTTP requests to the load balancer's IP to the `/predict/` endpoint with sample reviews.
    - ![image](https://github.com/user-attachments/assets/5698aa75-eeed-415d-8d8b-95d88126df0b)


3. **Load Testing with Locust:**
    - Install locust load testing package:
        - `pip install locust`
    - Run the load_test.py code in the same gcpdeploy folder with the command `locust -f load_tets.py`
    - Open the Locust web interface running at `0.0.0.0:8089` and start the test with the load balancer's IP address.
    - <img width="897" alt="Screenshot 2024-12-04 at 2 05 35 AM" src="https://github.com/user-attachments/assets/c88180c3-b787-4b8d-908f-2a5166c39643">

4. **Monitor Auto-scaling:**
    
    - With the simulated load of 200 users and a spawn rate of 200 requests per second, observe scaling to 3 instances.
    - Monitor the instances and performance through the GCP Console to see if new instances are created and traffic is balanced.
    - After the load test, observe the scale-down behavior when the traffic decreases.
    - <img width="1465" alt="Screenshot 2024-12-04 at 2 08 11 AM" src="https://github.com/user-attachments/assets/2e1d1c97-3e1c-414a-86a6-22fa61ea66b5">

This step ensures that our application can handle varying loads by automatically scaling the number of instances and distributing the load effectively.

# Flask API - UI Page
The UI for our application has been developed using flask API served on port 8000. The UI after being hosted on the load balancer looks like this:
![image](https://github.com/user-attachments/assets/ee581f5e-89e6-4f5b-8bcc-91d049a0d014)

One the bank IT employee hits on Launch prediction tool, it will take them to the prediction page which will look like the one below:
![image](https://github.com/user-attachments/assets/f9aeccf0-40c9-4408-82c4-bfb3754ddd89)

The end user can enter real time values and get an inference instantaneously

# Logging and Monitoring

## Overview

We have implemented logging and monitoring using BigQuery and Google Cloud Monitoring to capture and analyze data from model predictions. When the user enters input features, the model predicts whether the customer is likely to subscribe to a term deposit. The input values, predicted output, and related metadata are logged to BigQuery and Cloud Monitoring for auditing, performance tracking, and error analysis.

## BigQuery

BigQuery is Google's serverless, highly scalable, and cost-effective multi-cloud data warehouse. It allows us to store, query, and analyze large datasets efficiently. In this project, BigQuery is used to log prediction data, including input features, predictions, response times, and system statuses.

### BigQuery Setup

**1. Create a Dataset**

  - Open the Google Cloud Console.
  - Navigate to BigQuery.
  - Click the three dots next to your project name and select Create Dataset.
  - Enter the following details under Dataset Info:
    -  Name: `model_metrics_dataset`
    -  Data Location: `US`
  - Click `Create Dataset`.
    
**2. Create a Table**

  - Under the dataset (model_metrics_dataset), click the three dots and select Create Table.
  - Enter the following details under Table Info:
    - Name: `metrics_log`
    - Data Location: `US`
  - Click `Create Table`.
  - Go to the Schema section, click Edit Schema, enable Edit as Text, and paste the following JSON:
    ```sh
        [
          {"name": "timestamp", "type": "timestamp", "mode": "NULLABLE"},
          {"name": "endpoint", "type": "STRING", "mode": "NULLABLE"},
          {"name": "input_data", "type": "STRING", "mode": "NULLABLE"},
          {"name": "prediction", "type": "STRING", "mode": "NULLABLE"},
          {"name": "response_time", "type": "FLOAT", "mode": "NULLABLE"},
          {"name": "status", "type": "STRING", "mode": "NULLABLE"}
        ]
    ```

**3. Add Permissions**

  - Navigate to IAM & Admin in the Google Cloud Console.
  - Locate the service account being used and click Edit.
  - Add the following roles:
      - BigQuery Data Editor
      - BigQuery Data Viewer
      - BigQuery User
      - Storage Object Viewer
      - Editor
  - Click Save.

![image](https://github.com/user-attachments/assets/c1e9cdf0-5439-4171-9099-c031d70f6752)
   
Once the setup is complete, prediction logs are automatically written to BigQuery when the UI runs on the VM instance.

## Google Cloud Monitoring

Google Cloud Monitoring is a powerful observability tool that provides visibility into system health and performance. It enables us to monitor resource usage, application metrics, and logs. In this project, Cloud Monitoring is used to:

  - Log prediction data for performance and debugging.
  - Track key metrics such as response times, API health, model decay and system status.

### Setting Up Google Cloud Monitoring

  - Open the Google Cloud Monitoring Console.
  - Create a new workspace and associate it with your project.
  - Navigate to Metrics Explorer to create charts for monitoring metrics like response time and system status.
  - Set up alerting policies to notify you of anomalies, such as high response times or failed predictions.

## Dashboard Details

The monitoring dashboard provides a visual representation of system performance:
<img width="1402" alt="image" src="https://github.com/user-attachments/assets/d6278639-bc2c-4214-bd14-16f3667f34b2">
<img width="1405" alt="image" src="https://github.com/user-attachments/assets/ee18dd04-e925-40a3-96a4-e97a470e387b">

**Response Time Trends:** Tracks the average response time for predictions. Alerts have been configured to notify the team if the response time exceeds 3 milliseconds, ensuring prompt action for performance bottlenecks.

**Model Decay - Prediction Status:** Logs the predictions along with their statuses and displays the distribution of the output classes. An alert is triggered if the class distribution falls below 0.3, indicating potential issues like model drift or imbalanced predictions.

**CPU Utilization:** Monitors the CPU usage of the VM instance hosting the model. 

With this setup, you can ensure the system's reliability and quickly debug issues when needed.

# CI/CD for Model Pipeline

This project utilizes GitHub Actions to implement Continuous Integration and Continuous Deployment (CI/CD) for the model pipeline. This automation ensures that any changes made to the main branch trigger the model pipeline, facilitating seamless updates and deployments.

## Overview

The CI/CD workflow is designed to:

- Automatically trigger the model training and evaluation process whenever changes are pushed to the main branch.
- Ensure that the latest version of the model is always tested and deployed, maintaining high code quality and performance.

# Cost Analysis

Cloud Computing Resources:
1. Google Cloud Platform (GCP) VM Instance: $7.99/month x 12 months = $95.88/year
2. Say at a given time 8 instances are running: 8 * $95.88 = $767.04/year
3. Storage for datasets and intermediate results: $0.02/GB x 100GB/month = $6/month x 3 months = $18.
4. Compute Instance for API (e.g., Flask): = 50$ Approximately

# Testing
Before you push your code to GitHub, it's crucial to ensure that it meets our quality standards, including formatting, security, and functionality. To facilitate this, we recommend the following steps using `pytest` and `pylint`. These tools help identify formatting issues, potential vulnerabilities, and ensure that your test suites pass.
Every time a code push is made to the branch of our choice, the unittest written for data pipeline will get triggered with the help of GitHUb CI/CD Pipelines ensuring that there is no errors to the previous files when any new push is done to the branch of our choice.

## Airflow Dags

Once your code for data pipeline modules is built successfully, copy them to dags/src/. Create your Python Operator in airflow.py within dags/src/. Set pipeline dependencies using the >> operator.

After this step, we then proceed to edit our docker-compose.yaml file

Install and set up a docker desktop for building custom images from docker-compose.yaml file.

## Docker
Additional: If your code has extra dependencies, modify the docker-compose.yaml file. Add them under the Environment section or as follows:

Add your packages to _PIP_ADDITIONAL_REQUIREMENTS: in the docker-compose.yaml file.

Next, initialize the Airflow database as outlined in User Installation Step n. Then, continue with DAG development up to Step n.

If correctly set up, your module should appear in the DAG. If there are errors, you can check the logs and debug as needed.

## DVC Versioning

Setting up Data Versioning Control through dvc library installed as part of requirements.

1. Initialize dvc in the parent directory if not done already.
    ```python
    dvc init
    ```
2. Set up remote Google Cloud Storage connection.
    ```python
    dvc remote add -d myremote gs://<bucket>/<path>
    ```
3. Modify Google Cloud credentials to myremote by adding keys to the credential path.
    ```python
    dvc remote modify myremote credentialpath <GOOGLE-KEY-JSON-PATH>
    ```

## MLFlow

1. Establish the tracking URL for MLFlow:
```
mlflow.set_tracking_uri("ADD URL")
```
2. Setting the minimum logging level to record only warnings and more severe issues (errors and critical alerts):
```
logging.basicConfig(level=logging.WARN)
```
3. Set up the logger:
```
logger = logging.getLogger(__name__)
```

4. Optionally, you may or may not choose to ignore warnings:
```
warnings.filterwarnings("ignore")
```

