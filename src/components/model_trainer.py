import os
import sys
from catboost import CatBoostRegressor
from dataclasses import dataclass
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
import pandas as pd
from src.utils import save_object
from src.utils import evaluate_models
@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')
    
class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_array, test_array) -> float:
        """
        Train and evaluate multiple regression models with hyperparameter tuning,
        and save the best performing model.

        Args:
            train_array (numpy.ndarray): Training dataset (features and target combined).
            test_array (numpy.ndarray): Testing dataset (features and target combined).

        Returns:
            float: R² score of the best model on the test set.

        Raises:
            CustomException: If no model performs well on the datasets.
        """
       
        try:
            logging.info('Splitting training and test input data...')
            x_train, y_train = train_array[:, :-1], train_array[:, -1]
            x_test, y_test = test_array[:, :-1], test_array[:, -1]

            # Define models to evaluate (as in the original code)
            models_and_params = {
                    "Random Forest": {
                        "model": RandomForestRegressor(),
                        "params": {
                            "n_estimators": [50, 100, 200],
                            "max_depth": [None, 10, 20, 30],
                            "min_samples_split": [2, 5, 10]
                        }
                    },
                    "Gradient Boosting": {
                        "model": GradientBoostingRegressor(),
                        "params": {
                            "n_estimators": [100, 200, 300],
                            "learning_rate": [0.01, 0.1, 0.2],
                            "max_depth": [3, 5, 7]
                        }
                    },
                    "XGB Regressor": {
                        "model": XGBRegressor(),
                        "params": {
                            "n_estimators": [100, 200, 300],
                            "learning_rate": [0.01, 0.1, 0.2],
                            "max_depth": [3, 5, 7]
                        }
                    },
                    "CatBoost Regressor": {
                        "model": CatBoostRegressor(verbose=False),
                        "params": {
                            "iterations": [100, 200, 300],
                            "learning_rate": [0.01, 0.1, 0.2],
                            "depth": [3, 6, 10]
                        }
                    },
                    "AdaBoost Regressor": {
                        "model": AdaBoostRegressor(),
                        "params": {
                            "n_estimators": [50, 100, 150],
                            "learning_rate": [0.01, 0.1, 0.5, 1.0],
                            "loss": ['linear', 'square', 'exponential']
                        }
                    },
                    "Decision Tree": {
                        "model": DecisionTreeRegressor(),
                        "params": {
                            "max_depth": [None, 10, 20, 30],
                            "min_samples_split": [2, 5, 10],
                            "min_samples_leaf": [1, 2, 4]
                        }
                    },
                    "K-Neighbors Regressor": {
                        "model": KNeighborsRegressor(),
                        "params": {
                            "n_neighbors": [3, 5, 10, 15],
                            "weights": ['uniform', 'distance'],
                            "p": [1, 2]  # 1: Manhattan, 2: Euclidean
                        }
                    },
                    "Linear Regression": {
                        "model": LinearRegression(),
                        "params": {
                            "fit_intercept": [True, False],
                        }
                    }
                }

            # Evaluate models
            logging.info('Evaluating models...')
            best_model,evaluation_results= evaluate_models(
                x_train=x_train, y_train=y_train,
                x_test=x_test, y_test=y_test,
                models_and_params=models_and_params
            )

            # Save the best model
    
            save_object(self.model_trainer_config.trained_model_file_path, best_model)
            logging.info(f"Best model saved at {self.model_trainer_config.trained_model_file_path}")

            return evaluation_results

        except Exception as e:
            raise CustomException(e, sys)
