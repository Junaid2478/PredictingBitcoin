import pandas as pd
from math import sqrt
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
from math import ceil, sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import plotly.offline as py
import plotly.graph_objs as go
# splitting the data so it is 95% training and 5% test
from sklearn.model_selection import train_test_split


def run_knn():
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

    # scaling our data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['close'].values.reshape(-1, 1))
    print(scaled_data)

    # the max k values needed with our particular dataset, look into this
    print(f'MAX K VALUE TO SEARCH: {ceil(sqrt(data.shape[0]))}')
    print(data)
    print(type(data))

    X=data['date']
    y=data['close']

    data_np=np.array(data)
    X_train , X_test, y_train, y_test = train_test_split(X,y, test_size = 0.05, random_state = 0,shuffle=False)

    print(X_train.shape)

    # # normalising the dataset
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # scaled_y_train_data = scaler.fit_transform(y_train.values.reshape(-1, 1))

    # Created a PCA object that chooses the minimum number of components so that the variance is retained
    pca = PCA(.95)
    # finding the
    max_k_to_search=2
    rmse = []
    for K in range(max_k_to_search):
        K = K + 1
        KNN = KNeighborsRegressor(n_neighbors=K, weights = 'distance', p=1)
        X_train=np.array(X_train).reshape(-1, 1)

        X_train = np.array(range(len(X_train))).reshape(-1, 1)

        KNN.fit(X_train, y_train)  # fit the model
        X_test = np.array(X_test).reshape(-1, 1)
        predict = KNN.predict(X_test)  # make prediction on test set
        error = sqrt(mean_squared_error(y_test, predict))  # calculate rmse
        rmse.append(error)  # store rmse values
        print('RMSE value for k= ', K, 'is:', error)

    # visualizing the RMSE error depending on the K value

    # plt.figure(figsize=(16, 8))
    # plt.plot(range(1, 101), rmse, color='red', linestyle='dashed', marker='o',
    #          markerfacecolor='blue', markersize=10)
    # plt.title('Error Rate K Value')
    # plt.xlabel('K Value')
    # plt.ylabel('Mean Error')
    # plt.show()

    # This section graphs for a particular K value
    # https://www.diva-portal.org/smash/get/diva2:771141/FULLTEXT01.pdf
    # see conclusion about including all data values
    K=20**3
    test_size =len(y_test)
    X_steps=np.array(range(len(X)))

    X_train_steps=X_steps[: -test_size].reshape(-1, 1)

    y_train = y[: -test_size]

    X_test = X[-test_size:]
    X_test_steps=X_steps[-test_size:].reshape(-1, 1)

    y_test = y[-test_size:]

    KNN = KNeighborsRegressor(n_neighbors=K , weights = 'distance', algorithm='auto', p=2)

    # TODO change STEPS only
    y_train = np.array(y_train).reshape(-1, 1)

    KNN.fit(X_train_steps, y_train)
    X_test_dates = np.array(X_test).reshape(-1, 1)
    X_test = np.array(range(len(X_test_dates))).reshape(-1, 1)

    print(X_test)

    predict = KNN.predict(X_test_steps)  # make prediction on test set
    rmse = sqrt(mean_squared_error(y_test, predict))  # calculate rmse
    print(rmse)
    print(f'predict:{predict}')

    MAE = mean_absolute_error(y_test, predict)
    print(MAE)


    y_predicted  = predict.reshape(-1, 1)
    y_actual = np.array(y_test).reshape(-1, 1)

    X_test_shape = np.array(X_test)
    X = X_test_shape.flatten()
    Y =  y_actual.flatten()

    # plt.plot(data['date'], data['close'])
    fig = plt.figure(figsize=(16, 8))
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=20)
    plt.plot(X_test_dates.flatten(), Y, label = "Actual Bitcoin Price")
    plt.plot(X_test_dates.flatten(), y_predicted, label = "Predicted Bitcoin Price",  color='red', linestyle= 'dashed')

    return fig

"""
Because KNN looks at the N nearest points, and we are predicting the future, the same N nearest points will be considered each time. 
So, we need a very large N to avoid near-constant predictions.
As we are interpolating between the points, we can never predict large rises, especially beyond all-time highs in the training set.
Same goes for all time lows etc. 

This dataset is particularly tough for KNN, as is the problem of predicting the future rather than interpolation between known data points.

Overall, these show reasonable results in some instances, with poor results in general.
"""