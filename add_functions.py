#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.preprocessing import PolynomialFeatures,StandardScaler,MinMaxScaler,PowerTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV,train_test_split,TimeSeriesSplit
from sklearn import metrics
import re
import random
from statsmodels.tsa.seasonal import seasonal_decompose
from random import randrange
import scipy.stats as stats
import matplotlib.image as mpimg
import pmdarima as pm


# In[ ]:


def temperature_mean_change(data):
    hotter = {}
    unchanged = {}
    for city in list(data.City.unique()):
        # getting data, splitting into two sets
        city_data = data[data.City == city]
        city_1900 = city_data[(city_data.year >= 1900) & (city_data.year <= 1990)]
        city_2000 = city_data[(city_data.year >= 2000) & (city_data.year <= 2013)]
        # checking equality of variance:
        levene_result = stats.levene(city_1900.AverageTemperature,city_2000.AverageTemperature)[1]
        # setting parameters values for equal and unequal variances
        if levene_result < 0.05:
            equal_variance = False
        else:
            equal_variance = True
            
        t_result = stats.ttest_ind(city_1900.AverageTemperature,city_2000.AverageTemperature, equal_var = equal_variance)[1]
        if t_result < 0.05:
            hotter[city] = city_2000.AverageTemperature.mean() - city_1900.AverageTemperature.mean()
        else:
            unchanged[city] = city_2000.AverageTemperature.mean() - city_1900.AverageTemperature.mean()
    
    return hotter, unchanged


# In[ ]:


def translate_coordinates(data):
    for i in range(len(data.Longitude)):
        if data.Longitude.iloc[i][-1] == "E":
            try:
                data.Longitude.iloc[i] = float(data.Longitude.iloc[i][0:5])
            except:
                data.Longitude.iloc[i] = float(data.Longitude.iloc[i][0:4])
        elif data.Longitude.iloc[i][-1] == "W":
            try:
                data.Longitude.iloc[i] = -float(data.Longitude.iloc[i][0:5])
            except:
                data.Longitude.iloc[i] = -float(data.Longitude.iloc[i][0:4])
    for i in range(len(data.Latitude)):
        if data.Latitude.iloc[i][-1] == "N":
            try:
                data.Latitude.iloc[i] = float(data.Latitude.iloc[i][0:5])
            except:
                data.Latitude.iloc[i] = float(data.Latitude.iloc[i][0:4])
        elif data.Latitude.iloc[i][-1] == "S":
            try:
                data.Latitude.iloc[i] = -float(data.Latitude.iloc[i][0:5])
            except:
                data.Latitude.iloc[i] = -float(data.Latitude.iloc[i][0:4])
    return data


# In[ ]:


def find_best_order(orders,series):
    # creating variables
    best_order = ()
    best_score = float("inf")
    is_naive = ""
    # creating splits object
    X = series
    splits = TimeSeriesSplit(n_splits=4)
    for order in orders:
        # not all models can be created with given parameters
        try:
            print(order)
            # creating variables to save errors for each split
            errors = []
            naive_errors = []
            # iterating over splits
            for train_index,test_index in splits.split(X):
                # creating train and test variables
                train = X.iloc[train_index]
                test = X.iloc[test_index]

                # creating variables for one forward testing
                history = [x for x in train]
                predicted = []
                naive = X.iloc[test_index-1]

                # one forward arima testing
                for t in test:
                    # creating and fitting model
                    model = ARIMA(history,order)
                    model_fit = model.fit()
                    # forecasting and appending variables
                    val = model_fit.forecast()[0]
                    predicted.append(val)
                    history.append(t)

                # calculating and appending errors for current split
                error = metrics.mean_squared_error(test,predicted)
                naive_err = metrics.mean_squared_error(test,naive)
                errors.append(error)
                naive_errors.append(naive_err)

            # calculating mean prediction error for tested order
            error_order = np.mean(errors)
            naive_error = np.mean(naive_errors)

            # checking if error is smaller than threshold (naive prediction error)
            if error_order < naive_error:
                is_naive = "No"
                # checking if error is smaller than current best model error
                if error_order < best_score:
                    # updating variables
                    best_order = order
                    best_score = error_order
            # passing orders which couldn't be constructed
        except:
            continue
    # prediction is worse than naive baseline
    if is_naive == "Yes":
        return "No prediction was better than naive"
    # prediction is better than naive baseline
    elif is_naive == "No":
        return best_order

