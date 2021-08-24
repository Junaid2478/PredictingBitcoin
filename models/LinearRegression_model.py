import math

import pandas as pd
from sklearn import neighbors
from math import sqrt
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error

def run_linear_regression():
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
    train, test = train_test_split(data, test_size=0.05, shuffle=False)

    X_train = np.array(train['date'])
    y_train = np.array(train['close']).reshape(-1, 1)

    X_test = np.array(test['date'])
    y_test = np.array(test['close']).reshape(-1, 1)

    # scaling our data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['close'].values.reshape(-1, 1))
    print(scaled_data)

    # TODO: Replace all steps with label encoder
    label_encoder = LabelEncoder()
    all_data=np.append(X_train,X_test)
    print(all_data)
    X_steps = label_encoder.fit_transform(all_data)
    X_train_steps =X_steps[:len(X_train)].reshape(-1,1)
    X_test_steps =X_steps[len(X_train):].reshape(-1,1)
    # X_test_steps =  label_encoder.fit_transform(X_test).reshape(-1, 1)


    """
    Gives each string a label, the label is an integer that allows us to compute the distance between datapoints. this is needed for all models. 
    
    Alternatively, we could have used one hot encoding, but this also works and is simpler. (one hot is better for higher dimensional data, we have 2d data)
    
    "1 JAN 2017: 06:00" -> 0
    "1 JAN 2017: 07:00" -> 1
    "1 JAN 2017: 07:00" -> 2
    """

    # print(integer_encoded)

    # X_train_steps = np.array(range(len(X_train))).reshape(-1, 1)
    # X_test_steps = np.array(range(len(X_train), len(X_train) + len(X_test))).reshape(-1, 1)

    polyreg = make_pipeline(
        PolynomialFeatures(degree=2),
        LinearRegression()
    )
    polyreg.fit(X_train_steps, y_train)

    y_pred = polyreg.predict(X_test_steps)

    error = sqrt(mean_squared_error(y_test, y_pred))
    print(error)

    MAE = mean_absolute_error(y_test,y_pred)
    print(MAE)

    # return generate_df(y_test, y_pred, X_test)
    fig = plt.figure()
    fig = plt.figure(figsize=(16, 8))
    plt.plot(X_test, y_test.flatten(), label="Actual Bitcoin Price")
    plt.plot(X_test, y_pred, label="Predicted Bitcoin Price", color='red', linestyle= 'dashed')
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=20)
    return fig

