import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers.recurrent import LSTM
from keras.layers import Conv1D
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from keras.optimizers import RMSprop
from sklearn.preprocessing import MinMaxScaler

def nMAPE(target, actual):
    if (not len(actual) == len(target) or len(actual) == 0):
        return -1.0
    total = 0.0
    cons1 = 10
    for x in range(len(actual)):
        if (target[x] <= cons1):
            total += abs((actual[x] - abs(target[x])) / cons1)
        else:
            total += abs((actual[x] - target[x])/actual[x])
    return total / len(actual)*100

def nRMSE(target, actual):
    sum = 0.0
    for i in range(len(target)):
        a = target.mean()
        sum = sum + np.power((target[i] - actual[i]), 2)
    return np.sqrt(sum / len(target))

def data_preprocessing(data, lookback=24):
    scalerX=MinMaxScaler()
    scalerY = MinMaxScaler()
    new_data=[]
    train_len=len(data)-3*lookback
    print((train_len, len(data)))
    for i in range(0, len(data)-lookback):
        new_data.append(data[i:i+lookback])
    new_data1=np.array(new_data)

    print('===============================================')
    print(new_data1.shape)
    feature_set=new_data1[:train_len+lookback, :-1]
    feature_set = scalerX.fit_transform(feature_set)
    X_train = feature_set[:train_len, :]
    X_test=feature_set[train_len :, :]
    # target set data preparation
    target_set=new_data1[lookback:, -1]
    y_old=target_set[train_len-lookback:train_len]
    print((target_set.shape, y_old.shape))
    target_set=target_set.reshape(-1, 1)
    target_set=scalerY.fit_transform(target_set)
    y_train=target_set[:train_len]
    y_test=scalerY.inverse_transform(target_set[train_len:train_len+lookback])
    print('Y_test data printing ..................')
    print((new_data1.shape, feature_set.shape, target_set.shape))
    print((X_train.shape, y_train.shape, X_test.shape, y_test.shape))
    X_train_3D=np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test_3D=np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    return scalerY, X_train, y_train, X_test, y_test, y_old, X_train_3D, X_test_3D,

def BPNN_model(X_train, y_train, X_test, y_test, scaler):
    print('BPNN modelling starting ---------')
    model=Sequential()
    model.add(Dense(24, input_dim=X_train.shape[1], kernel_initializer='uniform', activation='relu'))
    model.add(Dense(24, kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(12, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss="mean_squared_error", optimizer="RMSprop", metrics=['accuracy'])
    model.summary()
    history=model.fit(X_train, y_train, batch_size=32, epochs=150, validation_split=0.10, verbose=0)
    y_actual=model.predict(X_test)
    # y_actual=y_actual.reshape(-1, 1)
    y_actual=scaler.inverse_transform(y_actual)
    mape=nMAPE(y_test, y_actual)
    rmse=nRMSE(y_test,y_actual)
    return y_actual, mape, rmse
