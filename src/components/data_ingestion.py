import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTranformation
from src.components.model_trainer import ModelTrainer
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')
    

class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()
        
    def intiate_data_ingestion(self):
        logging.info('Entered the data ingestion method or component')
        try:
            # Read the dataset
            df = pd.read_csv('data/stud.csv')
            logging.info('Dataset loaded into a DataFrame')
            
            # Create necessary directories
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info('Raw dataset saved to artifacts directory')

            # Perform train-test split
            logging.info('Train-test split initiated')
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            
            # Save train and test sets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info('Train-test datasets saved successfully')

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path
        except Exception as e:
            raise CustomException(e, sys)
        

if __name__ == "__main__":
    # Create an object of DataIngestion
    obj = DataIngestion()
    
    # Trigger data ingestion
    train_data, test_data = obj.intiate_data_ingestion()
    logging.info(f"Data ingestion completed. Train data path: {train_data}, Test data path: {test_data}")
    data_tranformation=DataTranformation()
    train_array,test_array,_=data_tranformation.initiate_data_tranformation(train_data,test_data)
    model_trainer=ModelTrainer()
    result=model_trainer.initiate_model_trainer(train_array,test_array)
    
    df = pd.DataFrame(result,columns=['Model', 'R2 Score'])
    print(df)

    
