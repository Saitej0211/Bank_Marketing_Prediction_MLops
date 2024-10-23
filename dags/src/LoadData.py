import os
import logging
import io
import pandas as pd
import pickle

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_FILE_PATH = os.path.join(PROJECT_DIR, "data", "processed","raw_data.csv")
PICKLE_FILE_PATH = os.path.join(PROJECT_DIR, "data", "processed", "raw_data.pkl")

#DEFINE A FUNCTION TO LOAD AND STORE THE DATA
def load_data(pickled_file_path = PICKLE_FILE_PATH):
    try:
        # SET UP LOGGING TO A FILE
        logs_dir = os.path.join(PROJECT_DIR, "logs")
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        
        #CREATE THE LOGGING DIRECTORY IF NOT CREATED
        log_file_path = os.path.join(logs_dir, 'load_data.log')
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            handlers=[logging.FileHandler(log_file_path, mode='w'),
                                      logging.StreamHandler()])
        
        logging.info("Project directory fetched successfully")

        with open(pickled_file_path, 'rb') as f:
            loaded_data = pickle.load(f)

        logging.info("Pickle file loaded successfully")

        #Convert the loaded Data to a Dataframe and store it for future use
        if isinstance(loaded_data,bytes):
            csv_file = io.StringIO(loaded_data.decode('utf-8'))
            df = pd.read_csv(csv_file,sep=';')

            df.to_csv(OUTPUT_FILE_PATH, index=False)
            
            logging.info("Data saved as CSV to the output file path folder")
        
        return True
    
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return False