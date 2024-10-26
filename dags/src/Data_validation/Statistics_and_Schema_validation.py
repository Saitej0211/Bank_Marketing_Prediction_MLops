# Install required packages for TensorFlow Data Validation and TensorFlow
!pip install tensorflow_data_validation
!pip install --upgrade tensorflow
!pip install tensorflow tensorflow-data-validation protobuf==3.20.*
!pip install scikit-learn

# Import necessary libraries
import tensorflow as tf
import tensorflow_data_validation as tfdv
import pandas as pd
from sklearn.model_selection import train_test_split
from util import add_extra_rows  # Custom utility for adding rows
from tensorflow_metadata.proto.v0 import schema_pb2

# Print the versions of TFDV and TensorFlow to confirm installation
print('TFDV Version: {}'.format(tfdv.__version__))
print('Tensorflow Version: {}'.format(tf.__version__))

# Load the dataset
df = pd.read_csv('raw_data.csv', skipinitialspace=True)

# Split the data into training and evaluation sets, 80-20 split without shuffling
train_df, eval_df = train_test_split(df, test_size=0.2, shuffle=False)

# Preview the training set
train_df.head()

# Preview the evaluation set
eval_df.head()

# Add extra rows to the evaluation set, if required by the project context
eval_df = add_extra_rows(eval_df)

# Display the last 4 rows of the evaluation set to confirm addition of rows
eval_df.tail(4)

# Generate statistics for the training set using TensorFlow Data Validation
train_stats = tfdv.generate_statistics_from_dataframe(train_df)

# Display the training set statistics
train_stats

# Visualize the statistics to gain insights into data distributions, missing values, etc.
tfdv.visualize_statistics(train_stats)

# Infer a schema from the training data, which will serve as a template for expected data structure and feature types
schema = tfdv.infer_schema(statistics=train_stats)

# Define valid values for 'month' and 'day_of_week'
valid_months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
valid_days_of_week = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
 
# Update the schema to set valid values for 'month' and 'day_of_week'
for feature in schema.feature:
    if feature.name == 'age':
        # Set valid range for 'age'
        feature.int_domain.min = 18  # Minimum valid age
        feature.int_domain.max = 100  # Maximum valid age
        # Set valency for 'age'
        feature.value_count.min = 1  # Minimum number of values (one value expected)
        feature.value_count.max = 1  # Maximum number of values (one value expected)
    elif feature.name == 'month':
        feature.ClearField('domain')  # Ensure no separate domain is set
        feature.string_domain.value[:] = valid_months  # Set valid months directly
        # Set valency for 'month'
        feature.value_count.min = 1
        feature.value_count.max = 1
    elif feature.name == 'day_of_week':
        feature.ClearField('domain')  # Ensure no separate domain is set
        feature.string_domain.value[:] = valid_days_of_week  # Set valid days directly
        # Set valency for 'day_of_week'
        feature.value_count.min = 1
        feature.value_count.max = 1
 
# Display the updated schema
tfdv.display_schema(schema)

# Display the inferred schema
tfdv.display_schema(schema)

# Generate statistics for the evaluation set to compare with training data
eval_stats = tfdv.generate_statistics_from_dataframe(eval_df)

# Compare training and evaluation statistics side-by-side to check for any anomalies or distribution shifts
tfdv.visualize_statistics(
    lhs_statistics=eval_stats, 
    rhs_statistics=train_stats, 
    lhs_name='EVAL_DATASET', 
    rhs_name='TRAIN_DATASET'
)

