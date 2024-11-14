import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Import the function to test
from dags.src.eda import perform_eda, OUTPUT_DIR

class TestPerformEDA(unittest.TestCase):

    @patch('pandas.read_pickle')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_perform_eda(self, mock_close, mock_savefig, mock_read_pickle):
        # Mock the DataFrame
        mock_df = pd.DataFrame({
            'y': ['yes', 'no', 'yes', 'no'],
            'contact': ['cellular', 'telephone', 'cellular', 'telephone'],
            'housing': ['yes', 'no', 'yes', 'no'],
            'loan': ['yes', 'no', 'no', 'yes'],
            'default': ['no', 'no', 'yes', 'no'],
            'month': ['jan', 'feb', 'mar', 'apr'],
            'age': [25, 30, 35, 40],
            'job': ['admin.', 'blue-collar', 'entrepreneur', 'management'],
            'marital': ['married', 'single', 'divorced', 'married'],
            'education': ['primary', 'secondary', 'tertiary', 'unknown']
        })
        mock_read_pickle.return_value = mock_df

        # Call the function
        perform_eda('dummy_path')

        # Check if the plots were saved
        expected_plots = [
            "deposit_distribution.png",
            "contact_method_distribution.png",
            "housing_loan_distribution.png",
            "personal_loan_distribution.png",
            "default_status_distribution.png",
            "month_distribution.png",
            "age_distribution.png",
            "job_distribution.png",
            "marital_status_distribution.png",
            "education_distribution.png",
            "correlation_heatmap.png"
        ]

        for plot in expected_plots:
            mock_savefig.assert_any_call(os.path.join(OUTPUT_DIR, plot))

        # Check if all plots were closed
        self.assertEqual(mock_close.call_count, len(expected_plots))

    @patch('pandas.read_pickle')
    def test_perform_eda_empty_dataframe(self, mock_read_pickle):
        mock_read_pickle.return_value = pd.DataFrame()

        with self.assertRaises(Exception):
            perform_eda('dummy_path')

    @patch('pandas.read_pickle')
    def test_perform_eda_missing_columns(self, mock_read_pickle):
        mock_df = pd.DataFrame({'y': ['yes', 'no']})
        mock_read_pickle.return_value = mock_df

        with self.assertRaises(KeyError):
            perform_eda('dummy_path')

    @patch('pandas.read_pickle', side_effect=FileNotFoundError)
    def test_perform_eda_file_not_found(self, mock_read_pickle):
        with self.assertRaises(FileNotFoundError):
            perform_eda('non_existent_path')

    @patch('pandas.read_pickle')
    @patch('matplotlib.pyplot.savefig', side_effect=IOError)
    def test_perform_eda_save_plot_error(self, mock_savefig, mock_read_pickle):
        mock_df = pd.DataFrame({
            'y': ['yes', 'no'],
            'contact': ['cellular', 'telephone'],
            'housing': ['yes', 'no'],
            'loan': ['yes', 'no'],
            'default': ['no', 'no'],
            'month': ['jan', 'feb'],
            'age': [25, 30],
            'job': ['admin.', 'blue-collar'],
            'marital': ['married', 'single'],
            'education': ['primary', 'secondary']
        })
        mock_read_pickle.return_value = mock_df

        with self.assertRaises(IOError):
            perform_eda('dummy_path')

if __name__ == '__main__':
    unittest.main()