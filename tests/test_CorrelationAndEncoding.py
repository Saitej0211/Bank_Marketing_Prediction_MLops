# File: test_encoding_and_correlation_analysis.py

import unittest
from unittest.mock import patch
import pandas as pd
from dags.src.data_preprocessing.correlation_analysis import correlation_analysis
from dags.src.data_preprocessing.encoding import encode_categorical_variables

class TestEncodingAndCorrelationAnalysis(unittest.TestCase):

    @patch('pandas.read_pickle')
    def test_encode_categorical_variables(self, mock_read_pickle):
        # Mock data
        mock_df = pd.DataFrame({'Category': ['A', 'B', 'A', 'C'], 'Value': [1, 2, 3, 4]})
        mock_read_pickle.return_value = mock_df

        # Test encoding function
        with patch('pandas.DataFrame.to_pickle') as mock_to_pickle, patch('pandas.DataFrame.to_csv') as mock_to_csv:
            encode_categorical_variables("mock_path.pkl")
            # Check encoding and saving functions were called
            mock_to_pickle.assert_called_once()
            mock_to_csv.assert_called_once()

    @patch('pandas.read_pickle')
    def test_correlation_analysis(self, mock_read_pickle):
        # Mock data with numeric columns
        mock_df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [4, 3, 2, 1], 'C': [2, 3, 4, 5]})
        mock_read_pickle.return_value = mock_df

        # Test correlation function
        with patch('pandas.DataFrame.to_pickle') as mock_to_pickle, patch('pandas.DataFrame.to_csv') as mock_to_csv, patch('matplotlib.pyplot.savefig') as mock_savefig:
            correlation_analysis("mock_path.pkl")
            # Check that correlation matrix and plot were saved
            mock_to_pickle.assert_called_once()
            mock_to_csv.assert_called_once()
            mock_savefig.assert_called_once()

if __name__ == "__main__":
    unittest.main()
