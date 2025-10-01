import logging
import os
import sys

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import logger
from src.utils.exception_handling import CustomException
from src.utils.utilities import save_object


class DataTransform:
    def __init__(self,
                 train_data_path,
                 test_data_path,
                 numerical_columns,
                 categorical_columns,
                 target_column,
                 preprocessor_path
                 ):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.target_column = target_column
        self.preprocessor_path = preprocessor_path

    def get_data_transformation_object(self):
        try:
            num_pipeline = Pipeline(
                [
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scalar', StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                [
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('scaler', OneHotEncoder(handle_unknown='ignore'))
                ]
            )

            logging.info(f"Numerical Columns: {self.numerical_columns}")
            logging.info(f"Categorical Columns: {self.categorical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, self.numerical_columns),
                    ('cat_pipeline', cat_pipeline, self.categorical_columns)
                ]
            )

            logging.info("Preprocessor Created")
            return preprocessor
        except Exception as e:
            error_msg = CustomException(e, sys)
            logging.error('Error occurred while making transformation object')
            logging.error(f'Transformation Error: {error_msg}')
            raise CustomException(e, sys)

    def initiate_data_transfer(self):
        logging.info('Data Transformation initiated')

        try:
            train_data = pd.read_csv(self.train_data_path)
            test_data = pd.read_csv(self.test_data_path)

            preprocessor = self.get_data_transformation_object()

            train_input_features_df = train_data.drop(
                columns=[self.target_column], axis=1)
            train_target_feature = train_data[self.target_column]

            test_input_features_df = test_data.drop(
                columns=[self.target_column], axis=1)
            test_target_feature = test_data[self.target_column]

            logging.info(
                'Applying preprocessing object on training and test dataframe')

            input_features_train_arr = preprocessor.fit_transform(
                train_input_features_df)
            input_features_test_arr = preprocessor.fit_transform(
                test_input_features_df)

            train_arr = np.c_[input_features_train_arr,
                              np.array(train_target_feature)]
            test_arr = np.c_[input_features_test_arr,
                             np.array(test_target_feature)]

            save_object(file_path=self.preprocessor_path,
                        obj=preprocessor)

            logging.info(f'saved the preprocessor with file_name:\
                [{self.preprocessor_path}]')

        except Exception as e:
            error_msg = CustomException(e, sys)
            logging.error(f'Error occurred at: [{error_msg}]')
            raise CustomException(e, sys)


def path_joiner(file_path):
    cwd = os.getcwd()
    return os.path.join(cwd, file_path)


def main():
    params_file = os.path.join(os.getcwd(), 'config', 'params.yaml')
    config = yaml.safe_load(open(params_file))

    data_config = config['data']
    artifacts_config = config['src']['artifacts']

    train_data_path = path_joiner(artifacts_config['train_data_path'])
    test_data_path = path_joiner(artifacts_config['test_data_path'])
    numerical_columns = data_config['numerical_columns']
    categorical_columns = data_config['categorical_columns']
    target_column = data_config['target_column']
    print(train_data_path)
    preprocessor_path = path_joiner(
        config['src']['preprocessor']['preprocessor_path'])

    data_ingestion = DataTransform(train_data_path=train_data_path,
                                   test_data_path=test_data_path,
                                   numerical_columns=numerical_columns,
                                   categorical_columns=categorical_columns,
                                   target_column=target_column,
                                   preprocessor_path=preprocessor_path
                                   )
    
    data_ingestion.initiate_data_transfer()


if __name__ == "__main__":
    main()