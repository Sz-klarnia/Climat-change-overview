#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
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

import shapefile as shp
import matplotlib.pyplot as plt
import seaborn as sns
import shapefile
import geopandas as gpd
from bokeh.models import *
from bokeh.plotting import *
from bokeh.io import *
from bokeh.tile_providers import *
from bokeh.palettes import *
from bokeh.transform import *
from bokeh.layouts import *
from scripts import *

# In[ ]:


def temperature_mean_change(data):
    """
    Function to calculate if mean temperatures have changed from XX century to XXI fentury.
    Functions takes DataFrame, performs operations and returns dicts of cities in which temperature is unchanged and changed
    """
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



def find_best_order(orders,series):
    """
    Function evaluating ARIMA models. Function takes list of ARIMA orders and Series of data to fit models
    """
    
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


def geodf_create_transform(df, long_col, lat_col, to_resize=None, resize=5):
    """
    Function takes a pandas DataFrame (df), names of columns with Latitude and Longitude properties,
    name of a columns to resize, if neceseary and resizing scale. It returns a GeoDataFrame with latitudes and longitudes
    translated into pseudo Mercator web projection, used by number of online maps, such as OpenStreetMap
    """
    # creating geodataframe from given df, geometry from long and lat cols
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[long_col], df[lat_col]), crs=("EPSG:4326"))

    # changing geographical projection
    gdf = gdf.to_crs("EPSG:3857")

    # separating x_coordinates from geometry points
    gdf["x_crds"] = [x for x in gdf[long_col]]
    for i in range(len(gdf.geometry)):
        gdf.x_crds[i] = list(gdf.geometry[i].coords)[0][0]

    # separating y_coordinates from geometry points
    gdf["y_crds"] = [y for y in gdf[lat_col]]
    for i in range(len(gdf.geometry)):
        gdf.y_crds[i] = list(gdf.geometry[i].coords)[0][1]

    # resizing column if neceseary
    if to_resize != None:
        gdf["resized"] = gdf[to_resize] / resize

    # return transformed dataframe
    return gdf


# In[3]:


def get_max_min(x_max, x_min, y_max, y_min):
    """Function takes maximal and minimal geographical coordinates and translates them into pseudo Mercator points
    for the purpose of specyfing the maps' range"""
    # list of x and y coordinates
    x_max_min = [x_max, x_min]
    y_max_min = [y_max, y_min]

    # creating shapely points
    extremes = gpd.points_from_xy(x_max_min, y_max_min)

    # creating geoseries
    extremes = gpd.GeoSeries(extremes, crs=("EPSG:4326"))

    # reprojecting points to pseudo mercator
    extremes = extremes.to_crs("EPSG:3857")

    # defining ranges of x an y
    x_range = sorted([list(extremes[0].coords)[0][0], list(extremes[1].coords)[0][0]])
    y_range = sorted([list(extremes[0].coords)[0][1], list(extremes[1].coords)[0][1]])

    return x_range, y_range


# In[4]:


def plot_bubble_map(df, source, radius_col, hover_tuples, x_range, y_range, title=None, leg_label=None, plt=False,color="Red"):
    """ Ploting a stationary bubble map to display some phenomenon geographical distribution
    Function requires:
    Pandas or GeoPandas DataFrame with data
    Bokeh specific ColumnDataSource
    List of HoverTuples for creating a Hover Tool for map
    x and y range of the map to display
    Optional:
    a title to be displayed
    Legend label for circle plot
    Plt - wether to automaticly plot object or just return a figure

    Function automatically plots and displays a map"""

    # specyfying map provider
    tile_provider = get_provider(OSM)

    # plotting figure
    plot = figure(
        title=title,
        match_aspect=True,
        tools='wheel_zoom,pan,reset,save',
        x_range=x_range,
        y_range=y_range,
        x_axis_type='mercator',
        y_axis_type='mercator'
    )

    # switching of grid
    plot.grid.visible = True

    # plotting circles representing fires
    c = plot.circle("x_crds", "y_crds", source=source, size=1, radius=radius_col, color=color, alpha=0.5,
                    legend=leg_label)

    # creating HoverTool to display fire properties when mouse hovers above points
    circle_hover = HoverTool(tooltips=[x for x in hover_tuples], mode='mouse', point_policy='follow_mouse',
                             renderers=[c])
    # rendering points, adding them to plot
    circle_hover.renderers.append(c)
    plot.tools.append(circle_hover)

    # adding open street map
    map_ = plot.add_tile(tile_provider)
    map_.level = 'underlay'

    # switching of axes
    plot.xaxis.visible = False
    plot.yaxis.visible = False

    if plt==True:
        show(plot)
        output_notebook()
    else:
        return plot
    
def one_forward_prediction(data,order):
    """Function performing one forward ARIMA prediction. Function takes data and order for ARIMA model"""
    prediction = {}
        
    model = ARIMA(data,order)
    model_fit = model.fit()
    fc,de, confint = model_fit.forecast(80)
    fc_idx = np.arange(data.index[-1],data.index[-1]+80)
    fc_series = pd.Series(fc,index=fc_idx)
    prediction["lower_cdi"] = pd.Series(confint[:, 0], fc_idx)
    prediction["upper_cdi"] = pd.Series(confint[:, 1], fc_idx)
    prediction["preiction"] = fc_series
    
    return prediction

def plot_forecast(historical_data,prediction_dict,title,xlabel,ylabel):
    """
    Function plots forecasts, showing historical data, prediction and confidence intervals for prediction
    """
    
    # plotting
    plt.show()
    plt.figure(figsize=(15,6))
    plt.plot(historical_data.index,historical_data,label = "historical data")
    plt.plot(prediction_dict["preiction"].index,prediction_dict["preiction"],label = "forecast")
    plt.fill_between(prediction_dict["lower_cdi"].index,
                prediction_dict["lower_cdi"],
                prediction_dict["upper_cdi"], color='k', alpha=.25, label = "95% confidence interval for forecast")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # limiting number of ticks showing on x axis
    plt.axes().xaxis.set_major_locator(plt.MaxNLocator(20))
    plt.axes().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.xticks(rotation=45)
    plt.legend(loc="upper left")
    plt.title(title)
    plt.show()

    
def one_forward_regression(data,target,model,X):
    """ Function takes training data, model and X values for which to make a prediction. 
        Function fits model on training data and predicts target value for X"""
    features = data.drop(target,axis=1)
    target = data[target]
    
    model.fit(features,target)
    result = model.predict(X)

    return result

def timeseries_split(data,x):
    """
    Function performs timeseries split along given point x"""
    
    try:
        train, test = data.loc[:x,:], data.loc[x+1:,:]
    except:
        train, test = data.loc[:x], data.loc[x+1:]
    return train, test


def walk_forward_validation(data,target,split,model):
    
    
    predictions = []
    naives = []
    # splitting data
    train,test = timeseries_split(data,split)
    
    history = train

    for i in range(len(test)):
        to_pred = test[test.index == test.iloc[i].name].drop(target,axis=1)
        # one forward prediction
        result = one_forward_prediction(history,target,model,to_pred)
        # appending history with the last test point
        history.append(test[test.index == test.iloc[0].name])
        # saving results in predictions
        predictions.append(result)
        
        # naive predition

        naives.append(test.iloc[i-1][target])

    
    naive_mse = mean_squared_error(test[target],naives)
    prediction_mse = mean_squared_error(test[target],predictions)
    
    if prediction_mse < naive_mse:
        return prediction_mse
    if prediction_mse > naive_mse:
        return "Prediction worse than naive"
    
def parts_per_million(data, co2_column, meth_column, n2o_column, year_col, year,land_emission_mean):
    """ formula to calculate rise in co2eq parts per million increase based on emissions"""
    co2 = data[data.year == year][co2_column].iloc[0] - data[data.year == year-100][co2_column].iloc[0] /2 + land_emissions_mean
    meth = data[data.year == year][meth_column].iloc[0] - data[data.year == year-10][meth_column].iloc[0]
    n2o = data[data.year == year][n2o_column].iloc[0] - data[data.year == year-100][n2o_column].iloc[0]
    
    ppm = (co2 + meth + n2o) / 10000000 / 7.82
    
    return ppm


def prepare_for_prediction(co2_trend,n2o_trend,ch4_trend,emissions_data, start_year,
                           end_year,concentrations,temp_data,land_use,just_co2=False):
    """Function prepares data to perform predictions on.
    co2_trend - trend for co2 emission
    n2o_trend - trend of n2o emission
    ch4_trend - trend for ch4 emission
    emissions_data - all data on emissions
    start_year - year in which predictios will start
    end_year - ending year of predictions
    concentration - concentration levels of greenhouse gasses
    temp_data - temperature levels
    just_co2 - False when calculating future levels of all gasses, true when calculating only co2 lvls
    Function returns prepared dataframe"""
    
    # concentration levels
    all_series = pd.Series(concentrations.all_gasses.values,index=concentrations.year)
    
    # range of years in prediction
    years = end_year-start_year
    
    # variable for storing co2 not in calculation
    co2_not = []
    # separating co2 to calculate and not in calculations
    if type(co2_trend) == list:
        co2_not = co2_trend[1]
        co2_trend = co2_trend[0]
        co2_future = pd.Series(np.linspace(co2_trend.max(),co2_trend.min(),years),index=np.arange(start_year,end_year))
        co2_future = co2_future + co2_not
    # creating future trend for emissions 
    else: 
        co2_future = pd.Series(np.linspace(co2_trend.max(),co2_trend.min(),years),index=np.arange(start_year,end_year))
    
    co2_future = co2_future.reset_index().rename({"index":"year"},axis=1)
    if just_co2 == True:
        n2o_future = n2o_trend.reset_index().rename({"index":"year"},axis=1)
        ch4_future = ch4_trend.reset_index().rename({"index":"year"},axis=1)
    else:
        n2o_future= pd.Series(np.linspace(n2o_trend.iloc[-1],0,years),index=np.arange(start_year,end_year)).reset_index().rename({"index":"year"},axis=1)
        ch4_future = pd.Series(np.linspace(ch4_trend.iloc[-1],0,years),index=np.arange(start_year,end_year)).reset_index().rename({"index":"year"},axis=1)
    
    # merging trends
    trend = co2_future.merge(n2o_future.merge(ch4_future,on="year"),on="year").rename({0:"co2","0_x":"nitrous_oxide","0_y":"methane"},axis=1)
    
    # merging with historical data
    all_years  = emissions_data.append(trend,ignore_index=True)
    all_years.drop_duplicates("year",inplace=True)
    all_years.index = all_years.year
    # specyfying increase of co2 concentration
    all_years["ppm_add"] = 0

    for i in range(start_year,end_year): 
        all_years["ppm_add"].loc[i] = parts_per_million(all_years,"co2","methane","nitrous_oxide","year",all_years["year"].loc[i],land_use)
    
    # adding concentration levels
    all_years["ppm"]= 0.0

    for i in all_series.index[1:]:
        all_years["ppm"][i] = all_series.loc[i]
    
    for i in range(start_year-1,end_year):
        all_years["ppm"].loc[i] = all_years["ppm"].loc[i-1] + all_years.ppm_add.loc[i]
        
    # creating data for prediction
    data_for_prediction = pd.DataFrame(all_years[["ppm"]])
    
    # adding temperature data
    data_for_prediction["temp"] = 0.0
    temp_series=pd.Series(data.AverageTemperature, index=data.index)

    for i in temp_series.index:
        data_for_prediction["temp"][i] = float(temp_series[i])
        
    # calculating previous year temperature
    data_for_prediction["temp_prev_year"] = 0.0

    for i in range(temp_series.index[1],end_year):
        data_for_prediction["temp_prev_year"][i] = data_for_prediction["temp"][i-1]
    
    data_for_prediction = data_for_prediction[data_for_prediction.ppm != 0]
    data_for_prediction = data_for_prediction.loc[1885:]

    
    return data_for_prediction
    
def translate_coordinates(data,lat_col,lon_col):
    """translating coordinates - removing direction letters, adding - and + where needed"""
    
    # translating coords
    North = [(x, float(data[lat_col][x][:-1])) for x in range(len(data[lat_col])) if data[lat_col][x][-1] == "N"]
    East = [(x, float(data[lon_col][x][:-1])) for x in range(len(data[lon_col])) if data[lon_col][x][-1] == "E"]
    South = [(x, -float(data[lat_col][x][:-1])) for x in range(len(data[lat_col])) if data[lat_col][x][-1] == "S"]
    West = [(x, -float(data[lon_col][x][:-1])) for x in range(len(data[lon_col])) if data[lon_col][x][-1] == "W"]
    
    # creating indexed Series 
    series_north = pd.Series([x[1] for x in North],index=[x[0] for x in North])
    series_south = pd.Series([x[1] for x in South],index=[x[0] for x in South])
    series_east = pd.Series([x[1] for x in East],index=[x[0] for x in East])
    series_west = pd.Series([x[1] for x in West],index=[x[0] for x in West])
    
    # defining series for each dimension
    Lat = series_south.append(series_north).sort_index()
    Long = series_east.append(series_west).sort_index()
    
    return Lat,Long
