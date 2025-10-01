import logging
import os
import sys
from dataclasses import dataclass

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.utils import logger
from src.utils.exception_handling import CustomException

# raw_data_file = path_joiner(data_config['data_file'])

# @dataclass
# class DataIngestionConfig:
#     processed_data_file = path_joiner(data_config['processed_data_path'])
#     training_data_file = path_joiner(data_config['train_data_path'])
#     testing_data_file = path_joiner(data_config['test_data_path'])

class DataIngestion:
    def __init__(self, raw_data_file, 
                 processed_data_file, 
                 training_data_file, 
                 testing_data_file,
                 random_state,
                 test_size):
        self.raw_data_file = path_joiner(raw_data_file)
        # self.data_saving_path = DataIngestionConfig()
        self.processed_data_file = processed_data_file
        self.training_data_file = training_data_file
        self.testing_data_file = testing_data_file
        self.random_state = random_state
        self.test_size = test_size
    
    def check_dir_exist(self, directory_path):
        dir_name = os.path.dirname(directory_path)
        os.makedirs(dir_name, exist_ok=True)
     
    
    def initiate_data_ingestion(self):
        logging.info('Data Ingestion Started')
        
        try:
            data = pd.read_csv(self.raw_data_file)
            
            train_data, test_data = train_test_split(data, 
                                                     random_state=self.random_state,
                                                     test_size=self.test_size)
            
            logging.info('Data Splitting finished')
            
            # artifacts_dir_name = os.path.dirname(self.processed_data_file)
            # os.makedirs(artifacts_dir_name, exist_ok=True)
            self.check_dir_exist(self.processed_data_file)
            self.check_dir_exist(self.training_data_file)
            self.check_dir_exist(self.testing_data_file)
            
            logging.info('Data Saving Started')
            
            data.to_csv(self.processed_data_file, index=False, header=True)
            train_data.to_csv(self.training_data_file, index=False, header=True)
            test_data.to_csv(self.testing_data_file, index=False, header=True)
            
            logging.info('Data Saving Finished!')
        
        except Exception as e:
            error_msg = CustomException(e, sys)
            logging.info(f"Error occured and error:[{error_msg}]")
            raise CustomException(e, sys)
        
def path_joiner(file_path):
    cwd = os.getcwd()
    return os.path.join(cwd, file_path)
    
def main():
    params_file = os.path.join(os.getcwd(),'config' ,'params.yaml')
    config = yaml.safe_load(open(params_file))

    data_config = config['data']
    artifacts_config = config['src']['artifacts']
    
    raw_data_file = path_joiner(sys.argv[1]) if sys.argv[1] else data_config['data_file']
    processed_data_file = path_joiner(artifacts_config['processed_data_path'])
    training_data_file = path_joiner(artifacts_config['train_data_path'])
    testing_data_file = path_joiner(artifacts_config['test_data_path'])
    random_state=data_config['random_state']
    test_size=data_config['test_size']
    
    data_ingestion = DataIngestion(
        raw_data_file= raw_data_file,
        processed_data_file= processed_data_file,
        training_data_file= training_data_file,
        testing_data_file=testing_data_file,
        random_state=random_state,
        test_size=test_size
    )
    
    data_ingestion.initiate_data_ingestion()
    
    
    

if __name__=="__main__":
    main()
            
            
