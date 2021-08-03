import math

import pandas as pd
from sklearn import neighbors
from math import sqrt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.offline as py

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import tensorflow as tf
from sklearn.model_selection import train_test_split

# read the dataset
data = pd.read_csv("C:/Users/jun/Documents/Computer Science/Semester 3/Dissertation/BTC-USD-New.csv")
dataset = data.values
data['date'] = pd.to_datetime(data['date'], format="%d/%m/%Y %H:%M")
# filters out data so we keep the date and close columns
data.drop('symbol', axis=1, inplace=True)
data.drop('tradecount', axis=1, inplace=True)
data.drop('high', axis=1, inplace=True)
data.drop('low', axis=1, inplace=True)
data.drop('unix', axis=1, inplace=True)
data.drop('Volume_USDT', axis=1, inplace=True)
data.drop('Volume_BTC', axis=1, inplace=True)
data.drop('open', axis=1, inplace=True)

# interpolates null values
data = data.interpolate(methods='linear')
# splitting the data so it is 95% training and 5% test

from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size=0.05)

X_train = list(range(len(train['date'])))
y_train = train['close']

X_test = list(range(len(test['date'])))
y_test = test['close']

# # normalising the dataset
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled_data = scaler.fit_transform(data['close'].values.reshape(-1, 1))
# print(scaled_data)
#
# # splitting the data so it is 95% training and 5% test
# training_data_len = int(len(scaled_data) * 0.95)
# test_data_len = len(scaled_data) - training_data_len
# train, test = scaled_data[0:training_data_len, :], scaled_data[training_data_len:len(scaled_data), :]
# print(len(train), len(test))
#
#
# # Created a PCA object that chooses the minimum number of components so that the variance is retained
# pca = PCA(.95)
# #
# # applying pca to our scaled data
# pca.fit(scaled_data)
# x_pca = pca.transform(scaled_data)

# create a random forrest object
model = RandomForestRegressor(n_estimators=10, random_state=0)
# look at regression model so it doesnt just look at next value
# fit the RFR with training data
model.fit(X_train.reshape(-1, 1), y_train(-1, 1)
# predict price
y_pred = model.predict(X_test.reshape(-1, 1))
print(f'predicted values: {y_pred}')

# evaluate using RMSE error

error = sqrt(mean_squared_error(y_test, y_pred))


# shows the magnitude of importance of features
# RandomForrest.save('C:/Users/jun/Documents/Computer Science/Semester 3/Dissertation/')
# use it to predict a certain date

# print(df['date'])

# real_chart = go.Scatter(x=data['date'], y=data['close'], name='Actual Bitcoin Price')
# forecast_chart = go.Scatter(x=data['date'], y=y_pred, name='Predicted Bitcoin Price')
# py.plot([real_chart, forecast_chart])
