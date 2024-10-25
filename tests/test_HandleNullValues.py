import unittest
from unittest.mock import patch, mock_open
import os

# Import the function to test
from dags.src.HandlingNullValues import process_data, PICKLE_FILE_PATH

class TestProcessData(unittest.TestCase):

    @patch('pandas.read_csv')
    @patch('pandas.DataFrame.to_pickle')
    @patch('os.path.exists')
    def test_file_creation(self, mock_exists, mock_to_pickle, mock_read_csv):
        # Mock DataFrame
        mock_df = mock_read_csv.return_value
        mock_df.isnull.return_value.mean.return_value = {}  # No columns to drop

        # Mock file existence check
        mock_exists.return_value = True

        # Call the function
        process_data()

        # Check if to_pickle was called with the correct file path
        mock_to_pickle.assert_called_once_with(PICKLE_FILE_PATH)

    @patch('pandas.read_csv')
    @patch('os.path.exists')
    def test_file_creation_exception(self, mock_exists, mock_read_csv):
        # Simulate an exception during file creation
        mock_read_csv.side_effect = Exception("Test exception")
        mock_exists.return_value = True

        # Call the function
        process_data()

        # Check that the file doesn't exist after an exception
        self.assertFalse(os.path.exists(PICKLE_FILE_PATH))

if __name__ == '__main__':
    unittest.main()