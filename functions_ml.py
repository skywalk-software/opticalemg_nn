# -*- coding: utf-8 -*-
"""
Created on Tues Mar  8 13:09:52 2022

@author: tyler 
"""

# %% Imports from this file
# from functions_ml import timeseries_from_sessions_list
# from functions_ml import apply_timeseries_cnn_v0
# from functions_ml import get_predictions

# %% Top-Level Imports
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from sklearn import preprocessing

# cnn model
from keras.models import Sequential
# from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.convolutional import AveragePooling1D
from tensorflow.keras.preprocessing import timeseries_dataset_from_array
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical


# %% FUNCTION - TAKES IN PREPROCESSED LIST OF SESSIONS, OUTPUTS A TIMESERIES DATASET
# TODO: Eliminate the timeseries issues from just concatenating the sessions together
# TODO: Verify that seq_length-1 causes prediction of latest timestep's value

def timeseries_from_sessions_list(sessions_list, seq_length, fit_scaler=False, scaler_to_use=None, imu_data=None):
    labels_array = np.empty((0,))
    data_array = np.empty((0, sessions_list[0]['skywalk'].shape[1]))
    for session in sessions_list:
        # Collapse labels onto skywalk timestamps
        labels_array = np.append(labels_array, np.array(session['contact'][0][session['skywalk'].index]), axis=0)
        partial_data_array = np.array(session['skywalk'])
        # add IMU data if applicable
        if imu_data is not None:
            for data_stream in imu_data:
                partial_data_array = np.append(partial_data_array, np.array(session[data_stream]), axis=1)
        data_array = np.append(data_array, np.array(session['skywalk']), axis=0)

    # Shifting indexing by seq_length-1 allows for prediction of the latest timestep's value
    # TODO shifting should occur on a per-array basis I believe -
    #  this shifting causes more issues at border between sessions
    data_array = data_array[:-(seq_length-1)]
    labels_array = labels_array[(seq_length-1):]

    # if fitting a new scaler
    if fit_scaler:
        if scaler_to_use is not None:
            raise ValueError(
                "Cannot assign scaler and fit a new one! Either change fit_scaler to False or remove scaler_to_use.")
        scaler = preprocessing.StandardScaler().fit(data_array)
        data_array = scaler.transform(data_array)
        out_dataset = tf.keras.preprocessing.timeseries_dataset_from_array(data_array,
                                                                           labels_array,
                                                                           sequence_length=seq_length,
                                                                           sequence_stride=1, sampling_rate=1,
                                                                           batch_size=32, shuffle=False, seed=None,
                                                                           start_index=None, end_index=None)
        return out_dataset, data_array, labels_array, scaler

    # If scaler was provided (e.g. this is test data)
    elif scaler_to_use is not None:
        data_array = scaler_to_use.transform(data_array)
        out_dataset = tf.keras.preprocessing.timeseries_dataset_from_array(data_array,
                                                                           labels_array,
                                                                           sequence_length=seq_length,
                                                                           sequence_stride=1, sampling_rate=1,
                                                                           batch_size=32, shuffle=False, seed=None,
                                                                           start_index=None, end_index=None)
        return out_dataset, data_array, labels_array

    # Default, no scaler at all
    else:
        out_dataset = tf.keras.preprocessing.timeseries_dataset_from_array(data_array,
                                                                           labels_array,
                                                                           sequence_length=seq_length,
                                                                           sequence_stride=1, sampling_rate=1,
                                                                           batch_size=32, shuffle=False, seed=None,
                                                                           start_index=None, end_index=None)
        return out_dataset, data_array, labels_array

# %% FUNCTIONS - v0 CNN model
def apply_timeseries_cnn_v0(train_dataset_internal, epochs, kernel_size, verbose=1):
    for data, labels in train_dataset_internal.take(1):  # only take first element of dataset
        numpy_data = data.numpy()
        numpy_labels = labels.numpy()
    batch_size, seq_length, n_features = numpy_data.shape[0], numpy_data.shape[1], numpy_data.shape[2]

    model = Sequential()
    # 1D convolution across time
    model.add(Conv1D(filters=64, kernel_size=kernel_size, activation='relu', input_shape=(seq_length, n_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(30, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    # Use categorical crossentropy for one-hot encoded
    # Use sparse categorical crossentropy for 1D integer encoded
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(train_dataset_internal, epochs=epochs, batch_size=batch_size, verbose=verbose)

    # print(model.summary())
    # print(model.evaluate(trainx, trainy))

    return model


# %% FUNCTIONS - v1 CNN model
def apply_timeseries_cnn_v1(train_dataset_internal, epochs, kernel_size, verbose=1):
    for data, labels in train_dataset_internal.take(1):  # only take first element of dataset
        numpy_data = data.numpy()
        numpy_labels = labels.numpy()
    batch_size, seq_length, n_features = numpy_data.shape[0], numpy_data.shape[1], numpy_data.shape[2]

    model = Sequential()
    # 1D convolution across time
    model.add(Conv1D(filters=48, kernel_size=kernel_size, activation='relu', input_shape=(seq_length, n_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(20, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    # Use categorical crossentropy for one-hot encoded
    # Use sparse categorical crossentropy for 1D integer encoded
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(train_dataset_internal, epochs=epochs, batch_size=batch_size, verbose=verbose)

    # print(model.summary())
    # print(model.evaluate(trainx, trainy))

    return model

#%% FUNCTION - GET PREDICTIONS
def get_predictions(model, test_dataset_for_pred):
    predictions_test = model.predict(test_dataset_for_pred)
    pred_array = np.array(predictions_test)
    gotten_predictions = np.zeros(pred_array.shape[0])
    for pred_i in range(pred_array.shape[0]):
        gotten_predictions[pred_i] = np.argmax(pred_array[pred_i, :])
    return gotten_predictions
