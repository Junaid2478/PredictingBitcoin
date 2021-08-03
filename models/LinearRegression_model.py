import math

import pandas as pd
from sklearn import neighbors
from math import sqrt
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import tensorflow as tf

from tensorboard import data

# read the dataset
data = pd.read_csv("C:/Users/jun/Documents/Computer Science/Semester 3/Dissertation/BTC-USD-New-Ordered.csv")
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
# replaces null values with average
data = data.interpolate(methods='linear')

X = data['date'].values.reshape(-1, 1)
y = data['close'].values.reshape(-1, 1)

# splitting the data so it is 95% training and 5% test
from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size=0.05)

X_train = list(range(len(train['date'])))
y_train = train['close']

X_test = list(range(len(test['date'])))
y_test = test['close']

model = LinearRegression
model.fit(X_train, y_train)

y_pred = model.predict(X_test.reshape(-1, 1))

print(model.coef_)