**Singapore Resale Flat Prices Predicting**


**Introduction :**

This project aims to construct a machine learning model and implement it as a user-friendly online application in order to provide accurate predictions about the resale values of apartments in Singapore. This prediction model will be based on past transactions involving resale flats, and its goal is to aid both future buyers and sellers in evaluating the worth of a flat after it has been previously resold. Resale prices are influenced by a wide variety of criteria, including location, the kind of apartment, the total square footage, and the length of the lease. The provision of customers with an expected resale price based on these criteria is one of the ways in which a predictive model may assist in the overcoming of these obstacles.


**Domain :** Real Estate

**Link1 :** LinkedIn Post / Working Model Video

**Prerequisites**

Python -- Programming Language
pandas -- Python Library for Data Visualization
numpy -- Fundamental Python package for scientific computing in Python
streamlit -- Python framework to rapidly build and share beautiful machine learning and data science web apps
scikit-learn -- Machine Learning library for the Python programming language.

Data Source
Link : https://beta.data.gov.sg/collections/189/view


**Project Workflow**

The following is a fundamental outline of the project:

The Resale Flat Prices dataset has five distinct CSV files, each representing a specific time period. These time periods are 1990 to 1999, 2000 to 2012, 2012 to 2014, 2015 to 2016, and 2017 onwards. Therefore, it is essential to merge the five distinct CSV files into a unified dataset.

The data will be converted into a format that is appropriate for analysis, and any required cleaning and pre-processing procedures will be carried out. Relevant features from the dataset, including town, flat type, storey range, floor area, flat model, and lease commence date will be extracted. Any additional features that may enhance prediction accuracy will also be created.

The objective of this study is to construct a machine learning regression model that utilizes the decision tree regressor to accurately forecast the continuous variable 'resale_price'.

The objective is to develop a Streamlit webpage that enables users to input values for each column and get the expected resale_price value for the flats in Singapore.


**Using the App**

Resale Price Prediction
To predict the resale price of a Singapore Flats, follow these steps:

Select the "Predictions" option menu.
Fill in the following required information:
Street Name
Block Number
Floor Area (Per Square Meter)
Lease Commence Date
Storey Range
Flat Type
Flat Model
Click the "PREDICT RESALE PRICE" button.
The app will display the predicted resale price based on the provided information.


**NOTE**

The "combined.csv" file used in Model.ipynb was froom data_cleaning_updated file and not from data_cleaning. Refer MRT and CBD distance from data_cleaning and  "combined.csv" from data_cleaning_updated.
