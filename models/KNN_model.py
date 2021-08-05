import pandas as pd
from math import sqrt
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
from math import ceil, sqrt
from sklearn.model_selection import train_test_split
import plotly.offline as py
import plotly.graph_objs as go
# splitting the data so it is 95% training and 5% test
from sklearn.model_selection import train_test_split

def new_dataset(dataset, time_step=1):
    """
    how many time steps to look back at to decide next data point
    :return:  Tuple of x values and y values
    """
    X_data, y_data = [], []
    for i in range(len(dataset) - time_step):
        x = dataset[i:(i + time_step), 0]
        X_data.append(x)
        y_data.append(dataset[i + time_step, 0])
    print(len(y_data))
    return np.array(X_data), np.array(y_data)

def split_dataset(scaled_data):
    """
    splitting the data so it is 95% training and 5% test
    :returns training set and test set as tuple
    """

    training_data_len = int(len(scaled_data) * 0.95)
    train, test = scaled_data[0:training_data_len, :], scaled_data[training_data_len:len(scaled_data), :]
    print(len(train), len(test))

    # generate dataset for train x, train y, test x and test y
    time_step = 1
    X_train, y_train = new_dataset(train, time_step)
    X_test, y_test = new_dataset(test, time_step)

    # applying pca to our algorithm
    # Created a PCA object that chooses the minimum number of components so that the variance is retained
    pca = PCA(.95)

    # applying pca to our scaled data
    pca.fit(scaled_data)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    # y_train = pca.fit_transform(y_train)
    # y_test = pca.transform(y_test)

    # reshaping our data for so it fits LSTM
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    return ((X_train,y_train),(X_test, y_test))



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

    # replaces null values with average
    data = data.interpolate(methods='linear')

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

    # https://www.analyticsvidhya.com/blog/2018/08/k-nearest-neighbor-introduction-regression-python/
    # visualizing the RMSE error depending on the K value

    # plt.figure(figsize=(16, 8))
    # plt.plot(range(1, 101), rmse, color='red', linestyle='dashed', marker='o',
    #          markerfacecolor='blue', markersize=10)
    # plt.title('Error Rate K Value')
    # plt.xlabel('K Value')
    # plt.ylabel('Mean Error')
    # plt.show()

    # This section graphs for a particular K value
    K=20**3
    test_size =len(y_test)
    X_steps=np.array(range(len(X)))

    X_train = X[: -test_size]
    X_train_steps=X_steps[: -test_size].reshape(-1, 1)

    y_train = y[: -test_size]

    X_test = X[-test_size:]
    X_test_steps=X_steps[-test_size:].reshape(-1, 1)

    y_test = y[-test_size:]

    KNN = KNeighborsRegressor(n_neighbors=K , weights = 'distance', algorithm='auto', p=2)
    X_train = np.array(X_train).reshape(-1, 1)

    # TODO change
    X_train = np.array(range(len(X_train),   )).reshape(-1, 1)

    y_train = np.array(y_train).reshape(-1, 1)

    KNN.fit(X_train_steps, y_train)
    X_test_dates = np.array(X_test).reshape(-1, 1)
    X_test = np.array(range(len(X_test_dates))).reshape(-1, 1)

    print(X_test)

    predict = KNN.predict(X_test_steps)  # make prediction on test set
    rmse = sqrt(mean_squared_error(y_test, predict))  # calculate rmse
    print(rmse)
    print(f'predict:{predict}')

    y_predicted  = predict.reshape(-1, 1)
    y_actual = np.array(y_test).reshape(-1, 1)

    X_test_shape = np.array(X_test)
    # real_chart = go.Scatter(x=X_test_shape, y=y_actual, name='Actual Bitcoin Price')
    # forecast_chart = go.Scatter(x=X_test_shape, y=np.array(y_predicted).transpose(), name=f'Predicted Bitcoin Price (rmse: {rmse})')

    X = X_test_shape.flatten()
    Y =  y_actual.flatten()

    # plt.plot(data['date'], data['close'])
    plt.plot(X_test_dates.flatten(), Y, label = "Actual Bitcoin Price")
    plt.plot(X_test_dates.flatten(), y_predicted, label = "Predicted Bitcoin Price",  color='red')
    plt.show()

