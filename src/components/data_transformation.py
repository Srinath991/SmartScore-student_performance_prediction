import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTranformatinConfig():
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTranformation():
    def __init__(self) -> None:
        self.data_tranformation_config = DataTranformatinConfig()
        
    def get_tranformer_object(self):
        """This function is responsible for data transformation

        Raises:
            CustomException: CustomException

        Returns:
            ColumnTransformer: column transformer object
        """
        try:
            categorical_features = [
            "gender", 
            "race_ethnicity", 
            "parental_level_of_education", 
            "lunch", 
            "test_preparation_course"
            ]

            numeric_features = [
            "reading_score", 
            "writing_score"
            ]

            # Numeric pipeline
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )
            logging.info('Numeric data standard scaling is completed')

            # Categorical pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )
            logging.info('Categorical data encoding is completed')

            # Column transformer to apply pipelines to columns
            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numeric_features),
                    ('cat_pipeline', cat_pipeline, categorical_features)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_tranformation(self, train_path, test_path):
        try:
            # Load the train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Read train and test data is completed')
            
            # Get preprocessor object
            logging.info('Obtaining preprocessor object')
            preprocessor_obj = self.get_tranformer_object()

            target_column = 'math_score'
            
            # Separate features and target
            input_features_train_df = train_df.drop(target_column, axis=1)
            target_features_train_df = train_df[target_column]
            
            input_features_test_df = test_df.drop(target_column, axis=1)
            target_features_test_df = test_df[target_column]
            
            logging.info('Applying preprocessing on train and test dataframes')
            
            # Apply transformations
            input_features_train_array = preprocessor_obj.fit_transform(input_features_train_df)
            input_features_test_array = preprocessor_obj.transform(input_features_test_df)
            
            # Combine input features and target
            train_arr = np.c_[
                input_features_train_array, target_features_train_df
            ]
            test_arr = np.c_[
                input_features_test_array, target_features_test_df
            ]
            logging.info('Saving preprocessing object')
            
            # Save the preprocessor object
            save_object(
                file_path=self.data_tranformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            
            return train_arr, test_arr, self.data_tranformation_config.preprocessor_obj_file_path
            
        except Exception as e:
            raise CustomException(e, sys)