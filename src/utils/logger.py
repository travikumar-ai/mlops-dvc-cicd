import logging
import os
from datetime import datetime

import yaml

config_yaml_file = os.path.join(os.getcwd(),'config' ,'params.yaml')

log_config = yaml.safe_load(open(config_yaml_file))['log']

log_path = os.path.join(os.getcwd(),log_config['log_path'])

os.makedirs(log_path, exist_ok=True)

file_name = f"{datetime.now().strftime('%d_%m_%y')}.log"
log_format = "[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s"

logging.basicConfig(
    filename = os.path.join(log_path,file_name),
    level = logging.INFO,
    format = log_format
    
)

if __name__ == "__main__":
    logging.info("Logging has started")
    logging.info("Logging has Finished")
    