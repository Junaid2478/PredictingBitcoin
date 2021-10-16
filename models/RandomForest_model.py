import math

import numpy as np
import pandas as pd
from math import sqrt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

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


    X_train_steps = np.array(range(len(X_train))).reshape(-1,1)
    X_test_steps = np.array(range(len(X_train),len(X_train) + len(X_test))).reshape(-1,1)

    # As this is a decision tree type of algorithm it does not require scaling

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
