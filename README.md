# Bank Marketing Campaign Prediction

This project aims to predict whether a client will subscribe to a term deposit based on direct marketing campaigns conducted by a Portuguese banking institution. The classification goal is to predict the outcome (`yes` or `no`) of the marketing campaign.

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
│   ├── test_preprocessing.py
│   ├── test_model.py
│   └── test_utils.py
├── Dockerfile              ## Docker container setup
├── docker-compose.yaml     ## Docker Compose for multi-container setups
└── requirements.txt        ## List of dependencies (Python libraries)

## How to Clone
1. **Clone the repository**:
   ```bash
   git clone https://github.com/Saitej0211/bank_marketing_prediction
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset
The dataset includes 45,211 instances and 16 features, covering multiple direct marketing campaigns. The goal is to classify whether a client will subscribe to a term deposit after receiving a phone call.

Dataset Link:
https://archive.ics.uci.edu/dataset/222/bank+marketing
