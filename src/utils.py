import os
import numpy as np
import pandas as pd
import sys
from src.exception import CustomException
import dill
from sklearn.metrics import r2_score
    
from sklearn.model_selection import GridSearchCV
from src.logger import logging
def save_object(file_path,obj):
    try:
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)


def evaluate_models(x_train, y_train, x_test, y_test, models_and_params, save_path="artifacts/model.pkl"):
    """
    Evaluate multiple models with hyperparameter tuning, return their performance scores,
    and save the best model to a file.

    Args:
        x_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training target.
        x_test (numpy.ndarray): Testing features.
        y_test (numpy.ndarray): Testing target.
        models_and_params (dict): Dictionary containing models and their hyperparameter grids.
        save_path (str): File path to save the best model.

    Returns:
        dict: A dictionary with model names as keys and their test R² scores as values.
    """
    try:
        report = {}
        best_model = None
        best_score = float('-inf')

        for model_name, model_data in models_and_params.items():
            logging.info(f"Starting hyperparameter tuning for {model_name}...")

            # Extract model and parameter grid
            model = model_data["model"]
            params = model_data["params"]

            # Perform hyperparameter tuning with GridSearchCV
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=params,
                scoring='r2',
                cv=5,
                verbose=1,
                n_jobs=-1
            )
            grid_search.fit(x_train, y_train)

            # Retrieve the best model from GridSearchCV
            best_estimator = grid_search.best_estimator_

            # Evaluate the best model on the test data
            y_test_pred = best_estimator.predict(x_test)
            test_r2_score = r2_score(y_test, y_test_pred)

            # Log the results
            logging.info(f"{model_name} achieved R² score: {test_r2_score:.4f} with best parameters: {grid_search.best_params_}")

            # Save the test R² score
            report[model_name] = test_r2_score

            # Update the best model if it outperforms the current best score
            if test_r2_score > best_score:
                best_score = test_r2_score
                best_model = best_estimator

        return best_model,report

    except Exception as e:
        raise CustomException(e, sys)

    
    
    
def load_object(file_path: str):
    try:
        with open(file_path, 'rb') as file:
            model = dill.load(file)
        return model
    except Exception as e:
        raise CustomException(f"Error loading model: {str(e)}", sys)



        
        
        