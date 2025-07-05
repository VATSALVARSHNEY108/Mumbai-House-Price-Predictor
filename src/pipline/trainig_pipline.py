import os 
import sys
import pandas as pd 
import numpy as np
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException

from src.components.data_ingestion import DataIngestion
from src.components.model_traning import ModelTraning
from src.components.data_transformation import DataTransformation
import warnings
warnings.filterwarnings("ignore")


# Run
if __name__=="__main__":
    obj = DataIngestion()
    train_data_path,test_data_path = obj.initated_data_ingestion()
    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.initatie_data_transformation(train_data_path, test_data_path)
    model_traning = ModelTraning()
    model_traning.initatied_model_traning(train_arr, test_arr)
