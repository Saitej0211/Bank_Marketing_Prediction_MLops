import os
import pandas as pd
import tensorflow as tf
import tensorflow_data_validation as tfdv
from tensorflow_data_validation.utils import slicing_util
from tensorflow_metadata.proto.v0.statistics_pb2 import DatasetFeatureStatisticsList
from generate_schema import generate_and_save_schema, load_data, prepare_train_data

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "processed")
INPUT_FILE_PATH = os.path.join(DATA_DIR, "raw_data.csv")
PICKLE_FILE_PATH = os.path.join(DATA_DIR, "processed_data.pkl")

def prepare_data_splits(df):
    train_len = int(len(df) * 0.7)
    eval_serv_len = len(df) - train_len
    eval_len = eval_serv_len // 2
    serv_len = eval_serv_len - eval_len

    train_df = df.iloc[:train_len].reset_index(drop=True)
    eval_df = df.iloc[train_len: train_len + eval_len].reset_index(drop=True)
    serving_df = df.iloc[train_len + eval_len: train_len + eval_len + serv_len].reset_index(drop=True)
    serving_df = serving_df.drop(['y'], axis=1)

    return train_df, eval_df, serving_df

def calculate_and_display_anomalies(statistics, schema):
    anomalies = tfdv.validate_statistics(schema=schema, statistics=statistics)
    tfdv.display_anomalies(anomalies=anomalies)

def visualize_slices_in_groups(sliced_stats, group_size=2):
    num_slices = len(sliced_stats.datasets)
    if num_slices == 0:
        print("No slices available in the sliced statistics.")
        return

    for i in range(0, num_slices, group_size):
        stats_list = []
        names_list = []
        for j in range(i, min(i + group_size, num_slices)):
            slice_stats_list = DatasetFeatureStatisticsList()
            slice_stats_list.datasets.extend([sliced_stats.datasets[j]])
            stats_list.append(slice_stats_list)
            names_list.append(sliced_stats.datasets[j].name)

        if len(stats_list) > 1:
            tfdv.visualize_statistics(
                lhs_statistics=stats_list[0],
                rhs_statistics=stats_list[1],
                lhs_name=names_list[0],
                rhs_name=names_list[1]
            )
            for k in range(2, len(stats_list)):
                tfdv.visualize_statistics(
                    lhs_statistics=stats_list[k-1],
                    rhs_statistics=stats_list[k],
                    lhs_name=names_list[k-1],
                    rhs_name=names_list[k]
                )
        else:
            tfdv.visualize_statistics(
                lhs_statistics=stats_list[0],
                lhs_name=names_list[0]
            )

def process_data(input_file_path, output_file_path):
    train_stats, schema, schema_file = generate_and_save_schema(input_file_path, os.path.dirname(output_file_path))

    df = load_data(input_file_path)
    train_df, eval_df, serving_df = prepare_data_splits(df)

    eval_stats = tfdv.generate_statistics_from_dataframe(eval_df)

    calculate_and_display_anomalies(eval_stats, schema=schema)

    options = tfdv.StatsOptions(schema=schema, infer_type_from_schema=True)
    serving_stats = tfdv.generate_statistics_from_dataframe(serving_df, stats_options=options)

    calculate_and_display_anomalies(serving_stats, schema=schema)

    schema.default_environment.extend(['TRAINING', 'SERVING'])
    tfdv.get_feature(schema, 'y').not_in_environment.append('SERVING')

    duration = tfdv.get_feature(schema, 'duration')
    duration.skew_comparator.infinity_norm.threshold = 0.03

    skew_drift_anomalies = tfdv.validate_statistics(train_stats, schema,
                                                    previous_statistics=eval_stats,
                                                    serving_statistics=serving_stats)
    tfdv.display_anomalies(skew_drift_anomalies)

    slice_fn = slicing_util.get_feature_value_slicer(features={'job': None, 'marital': None, 'education': None})
    slice_stats_options = tfdv.StatsOptions(schema=schema,
                                            experimental_slice_functions=[slice_fn],
                                            infer_type_from_schema=True)
    df.to_csv(input_file_path)
    sliced_stats = tfdv.generate_statistics_from_csv(input_file_path, stats_options=slice_stats_options)

    visualize_slices_in_groups(sliced_stats, group_size=3)

    df.to_pickle(output_file_path)

if __name__ == "__main__":
    process_data(INPUT_FILE_PATH, PICKLE_FILE_PATH)