# Bank Marketing Campaign Prediction
Sheela Hansda, Saiteja Reddy Gajula, Hashwanth Moorthy, Aishwarya Alagesan, Vignesh Sankar, Tejesvani Muppara vijayaram

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
│       ├── raw_data.pkl            ## Raw dataset in pickle format
│       ├── validate.csv            ## Data after data validation (CSV)
│       ├── validate.pkl            ## Data after data validation (Pickle)
│       ├── validate_process.csv    ## Data after anomaly detection (CSV)
│       └── validate_process.pkl    ## Data after anomaly detection (Pickle)
├── dags                            ## Directed Acyclic Graphs (DAGs) for pipeline orchestration
│   └── __init__.py                 ## Initialization file for DAGs module
├── logs                            ## Store log files during execution
│   ├── dag_id=DataPipeline         ## Logs related to DataPipeline DAG execution 
│   ├── dag_processor_manager       ## Logs related to DAG processor manager 
│   ├── download_data.log           ## Log for downloading data
│   ├── anomaly_detetcion.log       ## Log for anomaly detection
│   ├── data_schema_statistics_generation.log  ## Log for schema generation 
│   ├── eda.log                     ## Log for Exploratory Data Analysis 
│   └── outlier_handling.log        ## Log for outlier handling process 
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

![image](https://github.com/user-attachments/assets/0b752e97-0587-4bed-b5c8-c5883a71c745)

Pictured above: GCP Bucket tracking the Data Version for the dataset

# End-to-End Pipeline for Model Deployment
![ML Project Pipeline](https://github.com/user-attachments/assets/84b79898-c4cb-41bd-a07c-60d07986216d)

Pictured above: ML project Pipeline

## Pipeline Optimization
![image](https://github.com/user-attachments/assets/f669d1e1-7d8e-422b-b6ea-d5d5ceb0af1a)
Pictured above: Airflow DAG Execution Gantt Chart for Data Pipeline. It is a popular project management tool used to visualize and track the progress of tasks or activities over time. It provides a graphical representation of a pipeline's schedule, showing when each task is planned to start and finish.

Our Model Pipeline till this part for creating Data Pipeline has 2 major components
#### 1.Data Download
#### 2.Data Cleaning and Preprocessing

Our Pipeline begins with data acquisition where we fetch Data from the Source and perform Data Slicing. After the Data is downloaded, in the next step, we clean and Preprocess the Data. 
Next, the preprocessed data is analyzed to generate performance metrics.

We use Apache Airflow to orchestrate our data pipeline, treating each module as a distinct task within our primary DAG (Directed Acyclic Graph). This setup allows us to efficiently manage the flow from data acquisition to model deployment, ensuring each step is executed in the correct order and monitored for performance and success.

# Data Download
In this phase, the dataset is fetched and extracted into the designated data folder using the following modules:

- **DownloadData.py**: This python file downloads the latest available file from the GCP bucket and pickles it stores it in the folder.
- **LoadData.py**: This python file unpickles the data and stores it as a CCSV file in the data/processed folder allowing the next task to pick this up as input for data validation.

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
  
# Stats Gen
- **data_schema_statistics_generation.py**: Loads data from CSV file into a DataFrame and splits the data into training(60%), evaluation(20%), and serving(20%) datasets. Creates descriptive statistics for the training and serving datasets. Infers a training schema and serving schema(excluding the target variable Y) from the training statistics and saves it in .pbtxt format. Saves the data in .pkl format to be used for anomaly detection.
  
- **anomaly_detection.py**: Loads data from .pkl file and generates statistics to check for inconsistencies against a defined schema and identifies potential anomalies. Visualizations of data slices are created to highlight distribution differences. The data stored in .pkl file to be used for preprocessing.

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

### Model Performance Evaluation:

### Model Retraining

### Hyper Parameter Tuning

### Staging, Production and Archived models (MLFLOW)

### Logging and Monitoring

### Model Analysis

### Deployment Pipeline

### Cost Analysis

### Contributing / Development Guide

### Testing

### Step 1: Install Required Tools

### Step 2: Check Code Quality and Vulnerabilities

### Step 3: Run Test Suites


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

4. Optionally, you may or may not choose to ignore warnings:
```
warnings.filterwarnings("ignore")
```


