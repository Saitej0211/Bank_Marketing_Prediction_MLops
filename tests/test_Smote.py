import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import os

# Import the function to test
from dags.src.smote_analysis import smote_analysis, DATA_DIR

class TestSMOTEAnalysis(unittest.TestCase):

    @patch("dags.src.smote_analysis.pd.read_pickle")
    @patch("dags.src.smote_analysis.train_test_split")
    @patch("dags.src.smote_analysis.SMOTE")
    @patch("dags.src.smote_analysis.pd.DataFrame.to_csv")
    @patch("dags.src.smote_analysis.pickle.dump")
    def test_smote_analysis(self, mock_pickle_dump, mock_to_csv, mock_smote, mock_train_test_split, mock_read_pickle):
        # Mock DataFrame setup
        mock_df = mock_read_pickle.return_value
        mock_df.shape = (100, 3)  # Example shape with 100 rows and 3 columns
        mock_df.iloc = MagicMock()
        mock_df.iloc[:, :-1] = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100)
        })
        mock_df.iloc[:, -1] = pd.Series(np.concatenate(([0] * 90, [1] * 10)))  # Unbalanced target variable

        # Mock train-test split
        X_train, X_test = mock_df.iloc[:, :-1][:80], mock_df.iloc[:, :-1][80:]
        y_train, y_test = mock_df.iloc[:, -1][:80], mock_df.iloc[:, -1][80:]
        mock_train_test_split.return_value = (X_train, X_test, y_train, y_test)
        
        # Mock SMOTE behavior
        smote_instance = mock_smote.return_value
        X_train_resampled = np.random.rand(160, 2)  # After SMOTE, balanced dataset
        y_train_resampled = np.concatenate(([0] * 80, [1] * 80))
        smote_instance.fit_resample.return_value = (X_train_resampled, y_train_resampled)
        
        # Run smote_analysis
        with patch("builtins.open", mock_open()) as mock_file:
            train_pkl_path, test_pkl_path = smote_analysis("mock_input_path.pkl")

        # Assertions
        mock_read_pickle.assert_called_once_with("mock_input_path.pkl")
        mock_train_test_split.assert_called_once_with(mock_df.iloc[:, :-1], mock_df.iloc[:, -1], test_size=0.2, random_state=42)
        smote_instance.fit_resample.assert_called_once_with(X_train, y_train)
        
        # Verify CSV and pickle save calls
        mock_to_csv.assert_any_call(os.path.join(DATA_DIR, "smote_resampled_train_data.csv"), index=False)
        mock_to_csv.assert_any_call(os.path.join(DATA_DIR, "test_data.csv"), index=False)
        
        # Check pickle dump calls
        mock_pickle_dump.assert_any_call({'X_train': X_train_resampled, 'y_train': y_train_resampled}, mock_file())
        mock_pickle_dump.assert_any_call({'X_test': X_test, 'y_test': y_test}, mock_file())
        
        # Check returned paths
        self.assertEqual(train_pkl_path, os.path.join(DATA_DIR, "smote_resampled_train_data.pkl"))
        self.assertEqual(test_pkl_path, os.path.join(DATA_DIR, "test_data.pkl"))

if __name__ == '__main__':
    unittest.main()
