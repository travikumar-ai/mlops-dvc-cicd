import logging
import os
import pickle
import sys
from dataclasses import dataclass

import dagshub
import mlflow
import numpy as np
import pandas as pd
import yaml
from mlflow.models import infer_signature
from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_squared_error, r2_score,
                             root_mean_squared_error)
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from src.utils import logger
from src.utils.exception_handling import CustomException
from src.utils.utilities import save_object

# from src.utils.utilities import 


class ModelTrainer:
    def __init__(self, \
                 train_data_path, \
                 test_data_path, \
                 target_column, \
                 preprocessor_path,\
                 models_params,\
                 model_save_path \
                 ):
        
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.target_column = target_column
        self.preprocessor_path = preprocessor_path
        self.models_params = models_params
        self.model_save_path = model_save_path
        
    def evaluate_model(self, X_train, y_train, X_test, y_test, models:dict, params:dict):
        logging.info(f"Model evaluation started")

        try:
            train_report = {}
            test_report = {}
            models_best_params = {}
            
            for model_name, model in models.items():
                model_param = params[model_name]
                
                gs = GridSearchCV(model , model_param, cv=3)
                gs.fit(X_train, y_train)
                
                model.set_params(**gs.best_params_)
                model.fit(X_train, y_train) # Train model with best parameters

                best_params = gs.best_params_

                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                
                model_train_accuracy = r2_score(y_train, y_train_pred)
                model_test_accuracy  = r2_score(y_test, y_test_pred)
                
                train_report[model_name] = model_train_accuracy
                test_report[model_name] = model_test_accuracy
                models_best_params[model_name] = best_params
            logging.info(f"Model evaluation done!")
            return (train_report,
                    test_report,
                    models_best_params)
            
        except Exception as e:
            error_msg = CustomException(e, sys)
            logging.info(f"While evaluating Model error occurred at: {error_msg} ")
            raise CustomException(e, sys)
    
    
    def initiate_model_training(self):
        try:
            logging.info('model training started')
            
            with open(self.preprocessor_path, 'rb') as f:
                preprocessor = pickle.load(f)
            
            train_data = pd.read_csv(self.train_data_path)
            test_data = pd.read_csv(self.test_data_path)
            
            train_input_features_df = train_data.drop(columns=[self.target_column], axis=1)
            train_target_feature = np.array(train_data[self.target_column])
            
            test_input_features_df = test_data.drop(columns=[self.target_column], axis=1)
            test_target_features = np.array(test_data[self.target_column])
            
            train_arr = preprocessor.fit_transform(train_input_features_df)
            test_arr = preprocessor.transform(test_input_features_df)
            
            models = {
                'LinearRegression': LinearRegression(),
                'SVR': SVR(),
                'KNeighborsRegressor': KNeighborsRegressor(),
                'DecisionTreeRegressor': DecisionTreeRegressor(),
                'RandomForestRegressor': RandomForestRegressor(),
                
            }
            
            models_training_scores, models_test_scores, models_best_params = self.evaluate_model(X_train=train_arr,
                                                                                y_train=train_target_feature,
                                                                                X_test=test_arr,
                                                                                y_test=test_target_features,
                                                                                models=models,
                                                                                params= self.models_params
                                                                                )
            
            best_model_score = max(sorted(models_test_scores.values()))
            
            best_model_name = max(models_test_scores, key=models_test_scores.get)
            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")
            
            best_model = models[best_model_name]
            
            # Disable MLflow model registry calls for DagsHub compatibility
            os.environ["MLFLOW_ENABLE_MODEL_REGISTRY_SQL_BACKEND"] = "false"
            dagshub.init(repo_owner='travikumar3456', repo_name='mlops-dvc-cicd', mlflow=True)
            mlflow.set_experiment(experiment_id="1" )
            mlflow.autolog()
            with mlflow.start_run():
                best_params = models_best_params[best_model_name]
                
                best_model.set_params(**best_params)
                best_model.fit(train_arr, train_target_feature)
                
                mlflow.log_params(best_params)
                predicted = best_model.predict(test_arr)
                
                # Calculate and log metrics
                r2score = r2_score(test_target_features, predicted)
                mse = mean_squared_error(test_target_features, predicted)
                rmse = root_mean_squared_error(test_target_features, predicted)
                
                mlflow.log_metric("r2_score", r2score)
                mlflow.log_metric("mse", mse)
                mlflow.log_metric("rmse", rmse)
                
                mlflow.set_tag('model', best_model_name)
                
                # Infer signature correctly and log the model
                signature = infer_signature(train_arr, best_model.predict(train_arr))
                
                
                # mlflow.sklearn.log_model(
                #                 sk_model= best_model,
                #                 artifact_path='f23e9e88582d4457aa2ba81b220d7659',
                #                 signature=signature, 
                #                 # name='LRR'
                #             )
                
                
                
                # if best_model_score < 0.6:
                #     raise CustomException("No best model found", sys)
                logging.info(
                    f"Best found model on both training and testing dataset")
                
                save_object(
                    file_path=os.path.join(self.model_save_path,
                                           best_model_name + '.pkl'),
                    obj=best_model
                )
        except Exception as e:
            error_msg = CustomException(e, sys)
            logging.error('Error occurred while Training the model')
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
    model_save_path = path_joiner(
        config['src']['models']['model_save_path'])
    preprocessor_path = path_joiner(
        config['src']['preprocessor']['preprocessor_path'])
    
    models_params = config['models_params']
    

    model_trainer = ModelTrainer(train_data_path= train_data_path,
                                 test_data_path=test_data_path,
                                 target_column=target_column,
                                 preprocessor_path=preprocessor_path,
                                 models_params=models_params,
                                 model_save_path=model_save_path)
    
    model_trainer.initiate_model_training()


if __name__ == "__main__":
    main()