# import the libraries
import math

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import tensorflow as tf



# read the dataset
from tensorboard import data

df = pd.read_csv("C:/Users/jun/Documents/Computer Science/Semester 3/Dissertation/BTC-USD-New.csv")
print("data head = ", df.columns)

def plot_bitcoin_graph():
    date_strings_for_plot = df['date']

    df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d %H:%M:%S")

    # visualizes price of bitcoin
    df.index = df['date']

    plt.figure(figsize=(16, 8))
    plt.plot(df["close"], label='Close Price history')
    plt.show()


# plot_bitcoin_graph() visualizes graph
# df['date'] = pd.to_datetime(df['date'], format="%d/%m/%Y %H:%M")
# df['date'] = pd.to_numeric(pd.to_datetime(df['date']))

start = df.date

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

# replace each row
# removing the rows that are not needed
df.drop('symbol', axis=1, inplace=True)
df.drop('tradecount', axis=1, inplace=True)
df.drop('high', axis=1, inplace=True)
df.drop('low', axis=1, inplace=True)
df.drop('unix', axis=1, inplace=True)
df.drop('Volume_USDT', axis=1, inplace=True)
df.drop('Volume_BTC', axis=1, inplace=True)
df.drop('open', axis=1, inplace=True)

# Extracting date and close from the dataset
date = df.filter(['date']).values
close = df.filter(['close']).values

print("date = ", date)



# normalising the dataset
scaler = MinMaxScaler(feature_range=(0, 1))

date = scaler.fit_transform(date)
close = scaler.fit_transform(close)

scaled_date = preprocessing.scale(df)
scaled_close = preprocessing.scale(df)

# scaled_data = pd.DataFrame({'date': date[:, 0], 'close': close[:, 1]})

print("scaled data ", scaled_date)

# X represents the timestamp value, y represents the close value
X = date
y = close

# Separating dataset into training and testing values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# change null values to average
df.fillna(0, inplace=True)

print(df.shape)

# Created a PCA object that chooses the minimum number of components so that the variance is retained
# pca = PCA(.95)
#
# # calculates the loading scores and the variation each principle component accounts for
# pca.fit(dataset)
#
# pca_data = pca.transform(dataset)
#
# # Calculate the percentage of variation that each principle component accounts for
# per_variable = np.round(pca.explained_variance_ratio_ * 100, decimals=1)

# Implementing LSTM algorithm to predict closing price of bitcoin

# create a data frame with only close column
dataClose = df.filter(['close'])

datasetClose = dataClose.values

# gets the number of rows to train the model on (80%) while rounding the figure
training_size = math.ceil(len(dataset) * .8)

# our training data set is 29292
print(training_size)

# create the scaled training dataset
train_data = scaled_data[0: training_size, :]
# # spliting the dataset into trainX and trainY
trainX = []
trainY = []

prediction_days = 60


for i in range(prediction_days, len(train_data)):
    trainX.append(train_data[i - prediction_days:i, 0])
    trainY.append(train_data[i, 0])

    if i <= prediction_days:
        print(trainX)
        print(trainY)
        print()
# convert the x train and y train to numpy arrays
trainX, trainY = tf.convert_to_tensor(trainX), tf.convert_to_tensor(trainY)

# reshape the data the LSTM input expects the input to be 3 dimensional
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
trainX.shape

# building the LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(50, return_sequences=True, input_shape=(trainX.shape[1], 1)))
# more layers means more training, will overfit if too many layers
# We implement dropout regularization as mentioned in report to reduce overfitting
lstm_model.add(LSTM(50, return_sequences=False))
# We added a dense layer which would predict the next closing value
lstm_model.add(Dense(1))

# compile the model
# optimiser improves loss function, the loss function shows how well the model performed in training
lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# training the model
lstm_model.fit(trainX, trainY, batch_size=100, epochs=1)

lstm_model.save("C:/Users/jun/PycharmProjects/ML_models")

# how well the model would perform on the data we have, testing the model accuracy on existing data
# preparing the test data

test_data = scaled_data[training_size - 60:, :]

# creating x train and y test
testX = []
testY = scaled_data[training_size:, :]

for i in range(prediction_days, len(test_data)):
    print("this is run")
    testX.append(test_data[i - prediction_days: i, 0])

testX = tf.convert_to_tensor(testX)
# reshape the data so it works with the LSTM model as LSTM requires 3 dimensional data
print(testX.shape)
testX = tf.reshape(testX, (testX.shape[0], testX.shape[1], 1))

# Get the predicted values for the test X dataset
predictLstm = lstm_model.predict(testX)
predictLstm = scaler.inverse_transform(predictLstm)

# evaluating the model with RMSE
rmse = np.sqrt(np.mean(((predictLstm - testY) ** 2)))
# plotting our data
lstmPredicted = data[training_size]
lstmActual = data[training_size]
lstmActual['Predictions'] = predictLstm

#  visualizing the data
plt.figure(figsize=(16, 8))
plt.title('model')
plt.xlabel('date', fontsize=18)
plt.ylabel('close price USDT')
plt.plot(lstmActual['close'])
plt.plot(lstmPredicted[['close', 'predictions']])
plt.legend

# https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/