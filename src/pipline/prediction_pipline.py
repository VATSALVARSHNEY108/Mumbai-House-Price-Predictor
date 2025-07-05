import os
import sys 
import pandas as pd
import numpy as  np 
from src.logger import logging
from src.exception import CustomException

from src.utils import load_object


class PredictPipline:
    def __init__(self):
        pass

    def Predict(self,features):
        try:
            ## This line Of code Work in Any system
            preproccesor_path = os.path.join("artifcats","preprocessor.pkl")
            model_path = os.path.join("artifcats","model.pkl")

            preprocessor = load_object(preproccesor_path)
            model = load_object(model_path)

            data_scaled = preprocessor.transform(features)

            pred = model.predict(data_scaled)

            return pred

        except Exception as e:
            logging.info("Error Occured In Prediction Pipline")
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
           Area:int,
           Location:str,
           No_of_Bedrooms:float,
           New_Resale:int,
           Gymnasium:int,
           Lift_Available:int,
           Car_Parking:int,
           Maintenance_Staff:float,
           _24x7_Security:int,
           Childrens_Play_Area:int,
           Clubhouse:int,
           Intercom:int,
           Landscaped_Gardens:int,
           Indoor_Games:int,
           Gas_Connection:int,
           Jogging_Track:int,
           Swimming_Pool:int
           ):

        self.Area = Area
        self.Location = Location
        self.No_of_Bedrooms = No_of_Bedrooms
        self.New_Resale = New_Resale
        self.Gymnasium = Gymnasium
        self.Lift_Available = Lift_Available
        self.Car_Parking = Car_Parking
        self.Maintenance_Staff = Maintenance_Staff
        self._24x7_Security = _24x7_Security
        self.Childrens_Play_Area = Childrens_Play_Area
        self.Clubhouse = Clubhouse
        self.Intercom = Intercom
        self.Landscaped_Gardens = Landscaped_Gardens
        self.Indoor_Games = Indoor_Games
        self.Gas_Connection = Gas_Connection
        self.Jogging_Track = Jogging_Track
        self.Swimming_Pool = Swimming_Pool

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Area":[self.Area],
                "Location":[self.Location],
                "No_of_Bedrooms":[self.No_of_Bedrooms],
                "New_Resale":[self.New_Resale],
                "Gymnasium":[self.Gymnasium],
                "Lift_Available":[self.Lift_Available],
                "Car_Parking":[self.Car_Parking],
                "Maintenance_Staff":[self.Maintenance_Staff],
                "_24x7_Security":[self._24x7_Security],
                "Childrens_Play_Area":[self.Childrens_Play_Area],
                "Clubhouse":[self.Clubhouse],
                "Intercom":[self.Intercom],
                "Landscaped_Gardens":[self.Landscaped_Gardens],
                "Indoor_Games":[self.Indoor_Games],
                "Gas_Connection":[self.Gas_Connection],
                "Jogging_Track":[self.Jogging_Track],
                "Swimming_Pool":[self.Swimming_Pool],
            }

            data = pd.DataFrame(custom_data_input_dict)
            logging.info("DataFrame Gathered")
            return data

        except Exception as e:
            logging.info("Error Occured In Prediction Pipline")
            raise CustomException(e, sys)