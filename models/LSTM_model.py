# import the libraries
import math

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import plotly.offline as py
from sklearn.decomposition import PCA
import plotly.graph_objs as go
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout


def pre_processing():
    """
    Import data needed and process dataset for LSTM
    :return:  data df
    """
    data = pd.read_csv("C:/Users/jun/Documents/Computer Science/Semester 3/Dissertation/BTC-USD-New-Ordered.csv")
    print("data head = ", data.columns)
    dataset = data.values
    data['date'] = pd.to_datetime(data['date'], format="%d/%m/%Y %H:%M")

    data.drop('symbol', axis=1, inplace=True)
    data.drop('tradecount', axis=1, inplace=True)
    data.drop('high', axis=1, inplace=True)
    data.drop('low', axis=1, inplace=True)
    data.drop('unix', axis=1, inplace=True)
    data.drop('Volume_USDT', axis=1, inplace=True)
    data.drop('Volume_BTC', axis=1, inplace=True)
    data.drop('open', axis=1, inplace=True)

    print(data.shape)

    # visualizes the historic closing price of bitcoin
    data.index = data['date']

    # interpolates null values
    data = data.interpolate(methods='linear')

    # normalising the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['close'].values.reshape(-1, 1))
    print(scaled_data)

    return scaled_data, scaler, data

# I used this part to clean up the date attribute, there were multiple date formats
# and I only wanted one so I formatted the messy dates into 1
# POSSIBLE_DATE_FORMATS = ['%Y-%m-%d %I-%p', '%d/%m/%Y %H:%M:%S']
# for index, element in df['date'].iteritems():
#     #print("test1 i - ", i)
#
#     for date_format in POSSIBLE_DATE_FORMATS:
#
#         try:
#             parsed_date = datetime.strptime(element, date_format) # try to get the date
#             df['date'][index] = parsed_date
#
#             print("parsed date = ", parsed_date)
#
#             break # if correct format, don't test any other formats
#         except ValueError:
#             pass # if incorrect format, keep trying other formats

#  went through every column and replaced every date, to create a consistent pattern

# filters out data so we keep the date and close columns

# plt.figure(figsize=(16, 8))
# plt.plot(data["close"], label='Close Price history')
# plt.show()


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

    # reshaping our data for so it fits LSTM
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    return ((X_train,y_train),(X_test, y_test))

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

def build_model(trainset):
    """
    Build LSTM model
    :return: 
    """
    model = Sequential()
    X_train,_=trainset
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    # We implement dropout regularization as mentioned in report to reduce overfitting
    model.add(Dropout(0.15))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.15))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.15))
    model.add(Dense(units=1))  # prediction of the next closing value
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model

def train_model(model,train_test_datasets):
    ((X_train,y_train),(X_test, y_test))=train_test_datasets
    model.fit(X_train, y_train, epochs=25, batch_size=50, validation_data=(X_test, y_test), verbose=0, shuffle=False)

def predict(model,  testset):
    X_test,y_test=testset
    # making a prediction using testX and plotting it against test Y
    y_predict = model.predict(X_test)

    return (y_test,y_predict)

def get_rmse(y_actual, y_predict, scaler):
    # scaling back to normal values
    y_predicted_inverse = scaler.inverse_transform(y_predict.reshape(-1, 1))
    y_actual_inverse = scaler.inverse_transform(y_actual.reshape(-1, 1))

    rmse = np.sqrt(mean_squared_error(y_actual_inverse, y_predicted_inverse))
    print('Test RMSE: %.3f' % rmse)
    return rmse

def generate_graphs(data, y_actual, y_predict, X_test, rmse, scaler):
    # converting X column to dates
    x_dates = data .tail(len(X_test)).index
    print(f'LSTM predictDates: {x_dates}')

    # scaling back to normal values
    y_predicted_inverse = scaler.inverse_transform(y_predict.reshape(-1, 1))
    y_actual_inverse = scaler.inverse_transform(y_actual.reshape(-1, 1))

    # reshaping to be able to graph
    y_test_reshape = y_actual_inverse.reshape(len(y_actual_inverse))
    y_predict_reshape = y_predicted_inverse.reshape(len(y_predicted_inverse))

    real_chart = go.Scatter(x=x_dates, y=y_test_reshape, name='Actual Bitcoin Price')
    forecast_chart = go.Scatter(x=x_dates, y=y_predict_reshape, name=f'Predicted Bitcoin Price (rmse: {rmse})')
    py.plot([forecast_chart, real_chart])

def run_lstm():
    datascaled, scaler, data =pre_processing()
    train_test_datasets=split_dataset(datascaled)
    trainset, testset=train_test_datasets
    model=build_model(trainset)
    train_model(model,train_test_datasets)
    (y_test,y_predict)=predict(model, testset)
    rmse=get_rmse(y_test,y_predict, scaler)
    X_test,_=testset
    generate_graphs(data, y_test, y_predict, X_test, rmse, scaler)

