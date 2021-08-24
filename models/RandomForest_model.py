import math

import numpy as np
import pandas as pd
from math import sqrt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

def run_random_forest():
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

    # interpolates null values
    data = data.interpolate(methods='linear')

    # splitting the data so it is 95% training and 5% test
    train, test = train_test_split(data, test_size=0.05, shuffle=False)

    X_train = np.array(train['date'])
    y_train = np.array(train['close']).reshape(-1,1)

    X_test = np.array(test['date'])
    y_test = np.array(test['close']).reshape(-1,1)

    # label_encoder = LabelEncoder()
    # integer_encoded = label_encoder.fit_transform(X_train)
    # print(integer_encoded)

    X_train_steps = np.array(range(len(X_train))).reshape(-1,1)
    X_test_steps = np.array(range(len(X_train),len(X_train) + len(X_test))).reshape(-1,1)

    # X_train = list(range(len(train['date'])))
    # y_train = train['close']
    #
    # X_test = list(range(len(test['date'])))
    # y_test = test['close']

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


    # create a random forrest object
    model = RandomForestRegressor(n_estimators=100)

    # look at regression model so it doesnt just look at next value
    # fit the RFR
    model.fit(X_train_steps, y_train)

    # predict price
    y_pred = model.predict(X_test_steps)
    print(f'predicted values: {y_pred}')

    # evaluate using RMSE error
    error = sqrt(mean_squared_error(y_test, y_pred))
    print(error)

    MAE = mean_absolute_error(y_test, y_pred)
    print(MAE)

    fig = plt.figure()
    fig = plt.figure(figsize=(16, 8))
    plt.plot(X_test, y_test.flatten(), label="Actual Bitcoin Price")
    plt.plot(X_test, y_pred, label="Predicted Bitcoin Price", color='red', linestyle= 'dashed')
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=20)
    return fig



# https://neptune.ai/blog/random-forest-regression-when-does-it-fail-and-why
# Random Forrest suffers from the extrapolation problem when applied to our dataset and considering our dataset