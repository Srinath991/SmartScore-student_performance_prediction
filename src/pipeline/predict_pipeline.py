import pandas as pd
import sys
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self) -> None:
        pass
    def predict(self,features):
        try:
            model_path='artifacts/model.pkl'
            preprocessor_path='artifacts/preprocessor.pkl'
            
            model=load_object(model_path)
            preprocessor=load_object(preprocessor_path)
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)
    
class CustomData():
    def __init__(
        self,
        gender:str,
        race_ethnicity:str,
        parental_level_of_education:str,
        lunch:str,
        test_preparation_course:str,
        reading_score:int,
        writing_score:int,
        ) -> None:
       
        self.gender:str =gender
        self.race_ethnicity:str=race_ethnicity
        self.parental_level_of_education:str=parental_level_of_education
        self.lunch:str=lunch
        self.test_preparation_course:str=test_preparation_course
        self.reading_score:int=reading_score
        self.writing_score:int=writing_score
        
    def get_data_as_data_frame(self):
        try:
            student_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }
            return pd.DataFrame(student_dict)
        except Exception as e:
            raise CustomException(e, sys)

            

                
