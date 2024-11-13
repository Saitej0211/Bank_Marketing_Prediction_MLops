import unittest
import os
import pandas as pd
import logging
from unittest.mock import patch, MagicMock
from dags.src.data_preprocessing.datatype_format import process_datatype, handle_data_types
 
class TestProcessData(unittest.TestCase):
    
    @patch('os.path.exists')
    @patch('pandas.read_pickle')
    @patch('pandas.DataFrame.to_pickle')
    def test_process_datatype_success(self, mock_to_pickle, mock_read_pickle, mock_exists):
        # Mocking the file existence and data reading
        mock_exists.return_value = True
        
        # Create a sample DataFrame to return from the pickle file
        sample_data = pd.DataFrame({
            'numeric_col': [1, 2, 3],
            'string_col': ['A', ' B ', 'C'],
            'date_col': pd.to_datetime(['2023-01-01', '2023-01-02', None])
        })
        mock_read_pickle.return_value = sample_data
 
        # Call the process_datatype function
        output_path = process_datatype(input_file_path='dummy_path.pkl')
 
        # Check that the to_pickle was called
        mock_to_pickle.assert_called_once()
        self.assertIn('datatype_format_processed.pkl', output_path)
 
    @patch('os.path.exists')
    @patch('pandas.read_pickle')
    def test_process_datatype_file_not_found(self, mock_read_pickle, mock_exists):
        # Mocking the file existence to simulate a missing file
        mock_exists.return_value = False
        
        with self.assertRaises(FileNotFoundError):
            process_datatype(input_file_path='dummy_path.pkl')
 
    @patch('os.path.exists')
    @patch('pandas.read_csv')
    @patch('pandas.DataFrame.to_pickle')
    def test_process_datatype_csv(self, mock_to_pickle, mock_read_csv, mock_exists):
        # Mocking the file existence and data reading
        mock_exists.return_value = True
        
        # Create a sample DataFrame to return from the CSV file
        sample_data = pd.DataFrame({
            'numeric_col': [1, 2, 3],
            'string_col': ['A', ' B ', 'C'],
            'date_col': pd.to_datetime(['2023-01-01', '2023-01-02', None])
        })
        mock_read_csv.return_value = sample_data
 
        # Call the process_datatype function
        output_path = process_datatype(input_file_path='dummy_path.csv')
 
        # Check that the to_pickle was called
        mock_to_pickle.assert_called_once()
        self.assertIn('datatype_format_processed.pkl', output_path)
 
    @patch('os.path.exists')
    def test_handle_data_types(self, mock_exists):
        # Create a sample DataFrame
        sample_data = pd.DataFrame({
            'numeric_col': ['1', '2', '3'],
            'string_col': [' A ', ' B ', ' C '],
            'date_col': ['2023-01-01', '2023-01-02', None]
        })
 
        # Call handle_data_types function
        processed_data = handle_data_types(sample_data)
 
        # Assertions
        self.assertEqual(processed_data['numeric_col'].dtype, 'int64')
        self.assertEqual(processed_data['string_col'].dtype, 'string')
        self.assertEqual(processed_data['date_col'].dtype, 'datetime64[ns]')
 
    @patch('logging.info')
    def test_logging_info_called(self, mock_logging):
        # Call the function to test logging
        process_datatype(input_file_path='dummy_path.pkl')
        mock_logging.assert_any_call('Starting data processing')
 
if __name__ == '__main__':
    unittest.main()