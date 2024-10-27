# Bank Marketing Campaign Prediction
Sheela Hnasda, Saiteja Reddy Gajula, Hashwanth Moorthy, Aishwarya Alagesan, Vignesh Sankar, Tejesvani Muppara vijayaram

# Introduction

## Business Problem:

This competitive financial industry needs attraction and retention of clients, and one key offering of banks comes in the shape of term deposit accounts. These accounts lead to steady streams of revenue while developing client loyalty. To drive subscriptions, the bank will carry out direct marketing campaigns through calls, e-mails, and other means of contact with the clients. However, not all customers are equal, and as such, a significant amount of the marketing effort could be focused on those customers who have relatively lower interest in such products. 

As a result, this usually culminates in resource wastage, poor use of time, and suboptimal return on marketing investment. The challenge, therefore, is to identify which customers are most likely to subscribe based on historical data that will help the bank fine-tune its targeting strategy. A more focused approach would limit unproductive contacts, reduce costs, and enhance customer experience by concentrating on those who are more open toward term deposits, thereby creating a win for the bank and the clients.

## AI Solution Using Binary Classification Model:

To handle such a problem, the AI-driven binary classification model may be implemented that will label each client as likely to subscribe-yes or not-no based on a set of varied features. This model leverages historical campaign data, including client demographics such as age, income, and job type, their past interactions with the bank, and also broader socio-economic indicators such as employment rate and market conditions at the time of contact. With this, the model learns patterns and correlations that no human marketer would find. 

Hence, the model has learned what triggers increase the likelihood of subscription. When trained, it will then give an output for new clients with high accuracy; at this point, targeting by the bank can be restricted to those most likely to respond. This approach allows the bank to effectively enhance campaign efficiency, prioritize resources, and higher success rates of term deposit subscriptions, all while reducing overall cost and resources spent on less promising leads. The model also offers actionable insights into client behavior, thus enabling further refinement of future marketing strategies and deeper understanding of the client base.

The primary goal of this project is to develop a predictive model that accurately determines whether a client will subscribe to a term deposit based on past marketing campaign data. By using a binary classification approach, the project aims to optimize the bank's marketing strategy, improve the efficiency of client outreach, and increase the conversion rate of campaigns. This solution should ultimately enable the bank to make data-driven decisions that reduce costs, enhance customer engagement, and maximize return on investment for its marketing efforts.

# Dataset Information 
The dataset reflects the bank's direct marketing campaigns, where clients were primarily contacted by phone to promote term deposits. Since term deposits can require persuasion, many clients received multiple calls to determine their interest. Each call provided insights on the client's likelihood of subscribing ("yes") or declining ("no"), along with details on the timing and frequency of these interactions. By tracking how often and in what context clients responded positively, the data offers a detailed view of campaign dynamics, laying the groundwork for a predictive model to identify clients most likely to subscribe and thus enhance future campaign efficiency.

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
- **src/**: Source code for data preprocessing, feature engineering, and model training.
- **config/**: Configuration files for setting paths, parameters, and schemas.
- **pipeline/**: Scripts for running the end-to-end data pipeline.
- **tests/**: Unit tests for validating code functionality.
- **logs/**: Stores log files from pipeline runs.


## Git repository structure
```plaintext
.
├── LICENSE                 ## License information
├── README.md               ## Overview of the project, how to use, and dataset details
├── assets                  ## Store images, graphs, or other visualizations
├── config                  ## Configuration files for the project
│   ├── config.yaml         ## General configurations like file paths, hyperparameters
│   ├── constants.py        ## Constant variables (e.g., target variable, feature names)
│   └── schema.yaml         ## Dataset schema (columns and types)
├── data                    ## Raw and processed data files
│   ├── raw                 ## Raw dataset, e.g., bank-additional-full.csv
│   ├── processed           ## Processed data, train/test splits
├── logs                    ## Store log files during execution
│   └── training.log        ## Log details for model training
├── notebooks               ## Jupyter notebooks for exploratory data analysis (EDA)
│   ├── 01_EDA.ipynb        ## Exploratory Data Analysis notebook
├── pipeline                ## Pipeline scripts for orchestrating the workflow
│   ├── main.py             ## All-in-one script to run the entire pipeline
├── src                     ## Source code for the project
│   ├── __init__.py
│   ├── data_preprocessing  ## Data preprocessing scripts
│   │   ├── __init__.py
│   ├── feature_engineering ## Feature engineering scripts
│   │   ├── __init__.py
│   ├── models              ## Machine learning model scripts
│   │   ├── __init__.py
│   ├── utils               ## Utility functions used throughout the project
│   │   ├── __init__.py
├── tests                   ## Unit tests for various modules
│   └── test_utils.py
├── Dockerfile              ## Docker container setup
├── docker-compose.yaml     ## Docker Compose for multi-container setups
└── requirements.txt        ## List of dependencies (Python libraries)
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

6. Run the Docker composer.
   ```
   docker compose up
   ```
7. To view Airflow dags on the web server, visit https://localhost:8080 and log in with credentials
   ```
   user: airflow2
   password: airflow2
   ```
8. Run the DAG by clicking on the play button on the right side of the window

9. Stop docker containers (hit Ctrl + C in the terminal)
    
# Tools Used for MLOps

- GitHub Actions
- Docker
- Airflow
- Data Version Control (DVC)
- Google Cloud Storage (GCS)

## Google Cloud Platform (GCP)
We utilize Google Cloud Storage exclusively for storing our machine learning models, ensuring they are securely archived and readily accessible for deployment

One must set up a service account to use Google Cloud Platform services using below steps.

Go to the GCP Console: Visit the Google Cloud Console at https://console.cloud.google.com/.

Navigate to IAM & Admin > Service accounts: In the left-hand menu, click on "IAM & Admin" and then select "Service accounts".

Create a service account: Click on the "Create Service Account" button and follow the prompts. Give your service account a name and description.

Assign permissions: Assign the necessary permissions to your service account based on your requirements. You can either grant predefined roles or create custom roles.

Generate a key: After creating the service account, click on it from the list of service accounts. Then, navigate to the "Keys" tab. Click on the "Add key" dropdown and select "Create new key". Choose the key type (JSON is recommended) and click "Create". This will download the key file to your local machine.

You can avoid these steps of creating a GCP bucket, instead you could raise a request to access our GCP bucket

## Set up GCP Bucket and the Key
The dataset is downloaded and then uploaded to a GCP bucket which is alos tracked by DVC to account for any changes to the dataset. The Service account key needed to access the GCP bucket is created as saved under 'config/Key.json'. However, this is added to .gitignore to not be tracked by git. 

While setting up the project on your local system, please create the GCP bucket with your service account, replace the bucket name in the airflow.py file. Download the key associated with your service account and place it under config/Key.json and then run the code, it will work as expected.
