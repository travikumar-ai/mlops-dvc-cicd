import logging
import sys

from src.utils import logger


def ExceptionConfig(error, sys:sys):
    _, _, exc_tb = sys.exc_info()
    
    line_no = exc_tb.tb_lineno
    file_name = exc_tb.tb_frame.f_code.co_filename
    
    err_msg = ( f"Error occurred at file name [{file_name}]"
                f" at line no [{line_no}] error msg [{str(error)}])"
                )
    
    return err_msg

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = ExceptionConfig(
            error=error_message, sys=error_detail
        )

if __name__ == "__main__":
    logging.info("Logging has started")
    try:
        a = 1/0
    except Exception as e:
        logging.info('Trying to handle Exception')
        error_msg = CustomException(e, sys)
        logging.info(f'Handled Exception successfully and error is : {error_msg}')
