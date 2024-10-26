import os
import pandas as pd
import tensorflow_data_validation as tfdv

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "processed")
INPUT_FILE_PATH = os.path.join(DATA_DIR, "raw_data.csv")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")

def load_data(file_path):
    """Load CSV data into a dataframe."""
    return pd.read_csv(file_path, header=0, na_values='?')

def prepare_train_data(df):
    """Prepare training data from the full dataset."""
    train_len = int(len(df) * 0.7)
    return df.iloc[:train_len].reset_index(drop=True)

def generate_train_stats(train_df):
    """Generate statistics from the training dataset."""
    return tfdv.generate_statistics_from_dataframe(train_df)

def infer_schema(train_stats):
    """Infer schema from the computed statistics."""
    return tfdv.infer_schema(statistics=train_stats)

def save_schema(schema, output_dir):
    """Save the schema to a file."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    schema_file = os.path.join(output_dir, 'schema.pbtxt')
    tfdv.write_schema_text(schema, schema_file)
    return schema_file

def generate_and_save_schema(input_file_path, output_dir):
    """Main function to generate and save the schema."""
    df = load_data(input_file_path)
    train_df = prepare_train_data(df)
    train_stats = generate_train_stats(train_df)
    schema = infer_schema(train_stats)
    schema_file = save_schema(schema, output_dir)
    return train_stats, schema, schema_file

if __name__ == "__main__":
    train_stats, schema, schema_file = generate_and_save_schema(INPUT_FILE_PATH, OUTPUT_DIR)
    print(f"Schema saved to: {schema_file}")
    print(f"Number of features in schema: {len(schema.feature)}")