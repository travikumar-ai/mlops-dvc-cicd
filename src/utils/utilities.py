import logging
import os
import pickle
import sys

from src.utils import logger
from src.utils.exception_handling import CustomException


def save_object(file_path, obj):
    logging.info(f"Saving Object [{file_path}] started")
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj=obj, file=file_obj)
            
        logging.info(f"Object dumping done with file name [{file_obj}]")
    except Exception as e:
        error_msg = CustomException(e, sys)
        logging.info(f"Error occurred while dumping the [{file_obj}] object ")
        raise CustomException(e, sys)
    
        
    