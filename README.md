# Student Exam Performance Prediction

## Overview

The **Student Exam Performance Prediction** project is designed to predict students' performance based on various factors such as gender, ethnicity, parental education level, lunch type, test preparation course, reading score, and writing score. It utilizes machine learning models to provide predictions and insights, helping educators or stakeholders understand potential student performance outcomes.

This project includes a FastAPI-based web application that allows users to input student data and get performance predictions. The app is powered by a machine learning model that has been trained on a dataset and can be deployed using FastAPI.

## Features

- **Web Interface**: The app provides an HTML form to input student data.
- **Machine Learning Pipeline**: It includes a pre-trained model that predicts the performance of students.
- **Dynamic Predictions**: Users can input student data via the web interface, and the model will predict the performance.
- **Hyperparameter Tuning**: The model is trained with hyperparameter tuning to ensure high accuracy.

## Technologies Used

- **FastAPI**: FastAPI for building the web application.
- **Scikit-learn**: For model training, hyperparameter tuning, and evaluation.
- **XGBoost, CatBoost, Random Forest**: For machine learning models.
- **Pandas**: For data manipulation and preparation.
- **HTML/CSS**: For the frontend interface.

## Installation Instructions

Follow the steps below to set up the project locally.

### Prerequisites

- Python 3.7 or later
- Git
- Anaconda or Python virtual environment (optional, but recommended)