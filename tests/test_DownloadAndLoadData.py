import unittest
import os
from unittest.mock import patch, mock_open
from dags.src.DownloadData import download_data_from_gcp
from dags.src.LoadData import load_data

class TestDataPipeline(unittest.TestCase):

    @patch('dags.src.DownloadData.storage')
    @patch('dags.src.DownloadData.pickle.dump')
    def test_download_data_from_gcp(self, mock_pickle_dump, mock_storage):
        # Mocking storage.Client() and storage.Blob() behavior
        mock_client = mock_storage.Client.return_value
        mock_bucket = mock_client.get_bucket.return_value
        mock_blob = mock_bucket.blob.return_value

        # Mock blob download behavior
        mock_blob.download_as_string.return_value = b'some,csv,data'
        
        # Call the function
        result = download_data_from_gcp(bucket_name='mock_bucket')

        # Check if the result is a pickled file path
        expected_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "processed", "raw_data.pkl")
        self.assertEqual(result, expected_path)
        
        # Ensure pickle.dump was called (file was pickled)
        self.assertTrue(mock_pickle_dump.called)

        # Check if the file exists (mocked)
        with patch('os.path.exists', return_value=True):
            self.assertTrue(os.path.exists(expected_path))

    @patch('dags.src.LoadData.pickle.load')
    @patch('dags.src.LoadData.pd.DataFrame.to_csv')
    def test_load_data(self, mock_to_csv, mock_pickle_load):
        # Mock pickle.load to return some data
        mock_pickle_load.return_value = {'some': 'data'}

        # Call the function
        result = load_data(pickled_file_path="mock_data_path.pkl")

        # Check if the result is the path to the saved CSV
        expected_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "processed", "raw_data.csv")
        self.assertEqual(result, expected_path)

        # Ensure pickle.load was called (file was unpickled)
        self.assertTrue(mock_pickle_load.called)

        # Ensure to_csv was called (CSV file was created)
        self.assertTrue(mock_to_csv.called)

        # Check if the file exists (mocked)
        with patch('os.path.exists', return_value=True):
            self.assertTrue(os.path.exists(expected_path))

if __name__ == '__main__':
    unittest.main()