import os
import sys 
import pandas as pd
import numpy as  np 
from src.logger import logging
from src.exception import CustomException
from  dataclasses import dataclass

from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor,ExtraTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

from src.utils import save_object,model_evaluation



@dataclass
class ModelTraningConfig:
    train_model_file_obj = os.path.join("artifcats","model.pkl")


class ModelTraning:
    def __init__(self):
        self.model_traner_config = ModelTraningConfig()


    def initatied_model_traning(self,train_array,test_array):
        try:
            logging.info("Split Dependent And Independent Features")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "LinearRegression":LinearRegression(),
                "Ridge":Ridge(),
                "Lasso":Lasso(),
                "Elastic":ElasticNet(),
                "SVR":SVR(),
                "ExtraTreeRegressor":ExtraTreeRegressor(),
                "DecisionTreeRegressor":DecisionTreeRegressor(),
                "RandomForestRegressor":RandomForestRegressor()

            }
            params = {
                "LinearRegression":{
                    
                },
                "Ridge":{
                    "alpha": [0.01, 0.1, 1, 10,20]
                    
                },
                "Lasso":{
                    "alpha": [0.01, 0.1, 1, 10,20]
                },
                "Elastic":{
                    "alpha": [0.01, 0.1, 1, 10], "l1_ratio": [0.2, 0.4, 0.6, 0.8]
                },
                "SVR": {
                "gamma":["scale", "auto"],
                "C": [0.01, 0.1, 1, 10],
                },
                "ExtraTreeRegressor": {
                "criterion":["squared_error", "friedman_mse", "absolute_error", "poisson"],
                "splitter":['best','random'],
                "max_depth": [3, 5, 7, 9, 11],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features":["auto","sqrt","log2"],
                },
                "RandomForestRegressor":{
                    "criterion":["squared_error", "friedman_mse", "absolute_error", "poisson"],
                    'n_estimators': [ 180, 200,300],
                    'max_depth': [None, 5, 10,12,20],
                    'min_samples_split': [4, 5, 10,15,20],
                    'min_samples_leaf': [3, 5, 6,10,15],
                },
                "DecisionTreeRegressor":{
                    "criterion":["squared_error", "friedman_mse", "absolute_error", "poisson"],
                    "splitter":['best','random'],
                    "max_depth": [3, 5, 7, 9, 11],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features":["auto","sqrt","log2"]
                },
            }

            model_report:dict=model_evaluation(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                                models=models,param=params)

                ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))


            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            print(f"Best Model Found, Model Name is: {best_model_name},Accuracy_Score: {best_model_score}")
            print("\n***************************************************************************************\n")
            logging.info(f"Best Model Found, Model Name is: {best_model_name},Accuracy_Score: {best_model_score}")

            save_object(file_path=self.model_traner_config.train_model_file_obj,
                obj = best_model
                )

        except Exception as e:
            logging.info("Error Occured in Model Traning")
            raise CustomException(e,sys)