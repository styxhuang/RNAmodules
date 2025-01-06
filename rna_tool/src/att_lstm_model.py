import numpy as np
import pandas as pd
import random as rn
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Bidirectional
from keras.layers import LSTM
from attention import Attention

# attention、lstm计算
def att_model(structPath, num=312):
    # set random seed for attention
    rn.seed(num)
    tf.random.set_seed(num)
    model = Sequential()
    model.add(Attention())
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    data = np.array(structPath)
    data = data[:, 0:]
    [m1, n1] = np.shape(data)

    X = np.reshape(data, (-1, 1, n1))
    X = X.astype(np.float32) # RNA fold return int
    cv_clf = model
    tf.config.experimental_run_functions_eagerly(False)
    feature = cv_clf.predict(X)
    datadf = pd.DataFrame(data=feature)
    return datadf

def lstm_model(structPath, num=1846):
    # add random seed for deep fusion
    rn.seed(num)
    tf.random.set_seed(num)
    model = Sequential()
    model.add(Bidirectional(LSTM(200)))
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    data = np.array(structPath)
    data = np.nan_to_num(data, nan=-9999)
    data = data[:, 0:]
    [m1, n1] = np.shape(data)

    X = np.reshape(data, (-1, 1, n1))
    X = X.astype(np.float32) # RNA fold return int
    cv_clf = model
    tf.config.experimental_run_functions_eagerly(False)
    feature = cv_clf.predict(X)
    data_csv = pd.DataFrame(data=feature)
    return data_csv