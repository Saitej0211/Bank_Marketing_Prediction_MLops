import unittest
import os
import pandas as pd
import numpy as np
import logging
from unittest.mock import patch, MagicMock
from dags.src.data_preprocessing.outlier_handing import process_outlier_handling, handle_outliers
 
class TestOutlierHandling(unittest.TestCase):
    
    @patch('os.path.exists')
    @patch('pandas.read_pickle')
    @patch('pandas.DataFrame.to_pickle')
    def test_process_outlier_handling_success(self, mock_to_pickle, mock_read_pickle, mock_exists):
        # Mocking the file existence and data reading
        mock_exists.return_value = True
        
        # Create a sample DataFrame to return from the pickle file
        sample_data = pd.DataFrame({
            'A': [1, 2, 3, 100, 5],
            'B': [10, 15, 20, 25, -100],
            'C': [1, 2, 3, 4, 5]
        })
        mock_read_pickle.return_value = sample_data
 
        # Call the process_outlier_handling function
        output_path = process_outlier_handling(input_file_path='dummy_path.pkl')
 
        # Check that the to_pickle was called
        mock_to_pickle.assert_called_once()
        self.assertIn('outlier_handled_data.pkl', output_path)
 
        # Verify the handling of outliers
        handled_data = pd.read_pickle(output_path)
        self.assertTrue((handled_data['A'][3] <= 8) and (handled_data['B'][4] >= -10))
 
    @patch('os.path.exists')
    @patch('pandas.read_pickle')
    def test_process_outlier_handling_file_not_found(self, mock_read_pickle, mock_exists):
        # Mocking the file existence to simulate a missing file
        mock_exists.return_value = False
        
        with self.assertRaises(FileNotFoundError):
            process_outlier_handling(input_file_path='dummy_path.pkl')
 
    @patch('os.path.exists')
    def test_handle_outliers(self, mock_exists):
        # Create a sample DataFrame with outliers
        sample_data = pd.DataFrame({
            'numeric_col': [1, 2, 3, 100, 5, 6, -50],
            'another_col': [10, 15, 20, 25, 30, 1000, 40]
        })
 
        # Call handle_outliers function
        handled_data = handle_outliers(sample_data, threshold=1.5)
 
        # Assertions for outlier handling
        self.assertLessEqual(handled_data['numeric_col'][3], 8)  # Upper bound applied
        self.assertGreaterEqual(handled_data['numeric_col'][6], -12)  # Lower bound applied
        self.assertLessEqual(handled_data['another_col'][5], 37.5)  # Upper bound applied
 
    @patch('logging.info')
    def test_logging_info_called(self, mock_logging):
        # Call the function to test logging
        process_outlier_handling(input_file_path='dummy_path.pkl')
        mock_logging.assert_any_call('Starting outlier handling process')
 
if __name__ == '__main__':
    unittest.main()