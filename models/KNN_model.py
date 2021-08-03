import math

import pandas as pd
from sklearn import neighbors
from math import sqrt
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
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


# splitting the data so it is 95% training and 5% test
from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size=0.05)

X_train = list(range(len(train['date'])))
y_train = train['close']

X_test = list(range(len(test['date'])))
y_test = test['close']

# normalising the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_train = pd.DataFrame(X_train_scaled)
print(X_train)

x_test_scaled = scaler.fit_transform(X_test)
X_test = pd.DataFrame(x_test_scaled)

# Created a PCA object that chooses the minimum number of components so that the variance is retained
pca = PCA(.95)
#
# applying pca to our scaled data
pca.fit(X_train_scaled)
x_pca = pca.transform(X_train_scaled)

# looking at the error rate with different values of K and making predictions

rmse = []

local_minimum_error_pred=None
for K in range(1, 140):
    K = K + 1
    KNN = KNeighborsRegressor(n_neighbors=K)

    KNN.fit(X_train, y_train)  # fit the model
    predict = KNN.predict(X_test)  # make prediction on test set
    error = sqrt(mean_squared_error(y_test, predict))  # calculate rmse
    rmse.append(error)  # store rmse values
    print('RMSE value for k= ', K, 'is:', error)

    # from graph local minimum at K=70 (could use gradient descent etc to find it automatically)
    if K==70:
        local_minimum_error_pred=predict

# https://www.analyticsvidhya.com/blog/2018/08/k-nearest-neighbor-introduction-regression-python/


# visualizing the RMSE error depending on the K value
#
plt.figure(figsize=(16, 8))
plt.plot(range(1, 140), rmse, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()

