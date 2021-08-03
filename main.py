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

from models.LSTM_model import run_lstm
from models.KNN_model import run_knn


def main(modelname):
    if  modelname == 'LSTM':
        run_lstm()
    elif modelname=='KNN':
        run_knn()



if __name__=='__main__':
    model='KNN' # options are LSTM, KNN, RandForest, Linear
    main(model)
