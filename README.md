Traffic Volume Prediction – Machine Learning Project

This project predicts road traffic volume using machine learning techniques by analyzing multiple external factors such as weather conditions, temperature, road information, holidays, and time-based features. A trained model is deployed using a Flask web application to make real-time predictions.

Objective

The main goal is to predict hourly road traffic volume to help with:

Smart city monitoring

Traffic management & congestion prediction

Transport planning and road safety insights

Model Overview

The notebook traffic.ipynb performs:

Dataset preprocessing

Feature engineering (date-time features, holiday flags, weather mapping, etc.)

Model comparison and evaluation

Training and exporting the final selected model

The exported models (.joblib) are loaded inside the Flask app to make live predictions.

Features of Web Application
Feature	Description
User input form	Weather, temperature, holiday, road info, etc.
Real-time prediction	Predicts expected traffic volume instantly
Model transparency	Uses pre-trained .joblib model
Lightweight UI	Simple and intuitive to use

Dataset Sources

The dataset is a combination of:

Metro Interstate Traffic Volume

Weather dataset

Holiday dataset

Road information dataset

These multiple sources allow the model to learn real-world patterns affecting traffic.

Model Performance Summary

Based on experiments in the notebook:

Advanced traffic_model_app_pro.joblib shows higher accuracy

Feature engineering played a major role in improvement

Weather + Time + Road condition were the strongest predictors

Future Enhancements

Deployment on AWS/Render/Heroku

Live data streaming using sensors or APIs

Interactive dashboards with Grafana / Tableau / Power BI

Deep learning integration (LSTM for time-series)

