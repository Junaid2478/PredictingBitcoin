import pandas as pd
import sklearn.pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler

import numpy as np
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVR
import matplotlib.pyplot as plt
from math import  sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression




def run_ensemble():
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

    # replaces null values with interpolated data
    data = data.interpolate(methods='linear')

    X = data['date'].values.reshape(-1, 1)
    y = data['close'].values.reshape(-1, 1)

    # splitting the data so it is 95% training and 5% test
    train, test = train_test_split(data, test_size=0.05, shuffle=False)

    X_train = np.array(train['date'])
    y_train = np.array(train['close']).reshape(-1, 1)

    X_test = np.array(test['date'])
    y_test = np.array(test['close']).reshape(-1, 1)


    estimators = [
        ('lr', LinearRegression()),
        ('knn', KNeighborsRegressor(n_neighbors=8000, weights = 'distance', p=1)),
        ('rff', RandomForestRegressor())

    ]
    reg = StackingRegressor(
        estimators=estimators,
        final_estimator=(LinearSVR(random_state=42))
    )

    label_encoder = LabelEncoder()
    X_train_steps = label_encoder.fit_transform(X_train).reshape(-1, 1)
    X_test_steps =  label_encoder.fit_transform(X_test).reshape(-1, 1)

    reg.fit(X_train_steps, y_train)
    y_pred = reg.predict(X_test_steps)

    rmse= sqrt(mean_squared_error(y_test, y_pred))
    print(rmse)

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

    # lasso_pipeline
    #
    # randomforest_pipeline = sklearn.make_pipeline(tree_processor, RandomForestRegressor(random_state=42))
    # randomforest_pipeline
    #
    #
    #
    # estimators = [('Random Forest', randomforest_pipeline),
    #             ('Lasso', lasso_pipeline),
    #
    #
    # stacking_regressor = StackingRegressor(
    # estimators=estimators, final_estimator=RidgeCV())
    # stacking_regressor