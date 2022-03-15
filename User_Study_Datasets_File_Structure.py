# -*- coding: utf-8 -*-
"""
Created on Tues Mar  8 13:09:52 2022

@author: tyler 
"""

#%% Imports
import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
import random

from datetime import datetime
from os import listdir
from os.path import isfile, join
from tslearn.preprocessing import TimeSeriesResampler

import tensorflow as tf
from tensorflow import keras
# cnn model
from keras.models import Sequential
#from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.convolutional import AveragePooling1D
from keras.utils import to_categorical
from tensorflow.keras.preprocessing import timeseries_dataset_from_array
from tensorflow.keras import layers

#%% FUNCTION - CREATE USER OBJECT CLASS
# TODO: add methods for removal of a trial and automatic updating of num_trials
# TODO: add getNumSessions() method for getting total number of sessions? not sure if useful
# TODO: add padding function where you take multiple trials or sessions or shuffled events and concatenate them into a big array with padding so timeseries doesn’t get borked by a sudden edge

class User(object):
    def __init__(self, user_id, bio_data=dict.fromkeys(['name','birth_date','wrist_circumference','race_ethnicity','hairy_arms'])):
        self.user_id = user_id
        self.trials = []
        self.bio_data = bio_data
        self.trial_types = {}
        self.num_trials = 0
        
    def __repr__(self):
        return f'User(user_id="{self.user_id}") <bio_data={self.bio_data}, trial_types={self.trial_types}, num_trials={self.num_trials}>'

    def appendTrial(self, trial):
        if trial.user_id == self.user_id:
            if trial not in self.trials:
                self.trials.append(trial)
                self.num_trials += 1
                if trial.trial_type not in self.trial_types.keys():
                    self.trial_types[trial.trial_type] = 1
                else:
                    self.trial_types[trial.trial_type] += 1
            else:
                raise ValueError("Trial already in self.trials.", trial)
        else:
            raise ValueError("Trial and User user_id are not matched. Trial user_id:",trial.user_id,"User user_id:",self.user_id)
            
    # Clear all trials for this user
    def clearTrials(self):
        self.trials= []
        self.num_trials = 0
        self.trial_types = []
        
    # Set biograhical data for this user
    def setBioData(self, name=None, birth_date=None, wrist_circumference=None, race_ethnicity=None, hairy_arms=None):
        if name is not None: self.bio_data['name'] = name
        if birth_date is not None: self.bio_data['birth_date'] = birth_date
        if wrist_circumference is not None: self.bio_data['wrist_circumference'] = wrist_circumference
        if race_ethnicity is not None: self.bio_data['race_ethnicity'] = race_ethnicity
        if hairy_arms is not None: self.bio_data['hairy_arms'] = hairy_arms
    
    # Clear all biograhical data for this user
    def clearBioData(self):
        self.bio_data = dict.fromkeys(['name','birth_date','wrist_circumference','race_ethnicity','hairy_arms'])
        
    # Returns a list containing all trials that satisfy the search criteria. Adding no criteria is the same as calling user.trials
    def getTrials(self, trial_type=None, date=None, firmware_version=None, hand=None, notes=None):
        trials_list = self.trials
        
        if trial_type is not None:
            new_trials_list = []
            for trial in trials_list:
                if trial.trial_type == trial_type:
                    new_trials_list.append(trial)
            trials_list = new_trials_list
            
        if date is not None:
            if type(date) is str:
                date = datetime.strptime(date, '%Y-%m-%d').date()
            new_trials_list = []
            for trial in trials_list:
                if trial.date == date:
                    new_trials_list.append(trial)
            trials_list = new_trials_list
        
        if firmware_version is not None:
            new_trials_list = []
            for trial in trials_list:
                if trial.firmware_version == firmware_version:
                    new_trials_list.append(trial)
            trials_list = new_trials_list
            
        if hand is not None:
            new_trials_list = []
            for trial in trials_list:
                if trial.hand == hand:
                    new_trials_list.append(trial)
            trials_list = new_trials_list
            
        if notes is not None:
            new_trials_list = []
            for trial in trials_list:
                if trial.notes == notes:
                    new_trials_list.append(trial)
            trials_list = new_trials_list
            
        return trials_list
    
    # Returns a list containing all requested sessions
    # user_id.getSessions(user_id.getTrials(), how_many_sessions_list = [None, None, 1], second_half=False)
    def getSessions(self, trials_list, how_many_sessions_list=None, second_half=False):
        sessions_list = []
        for i in range(len(trials_list)):
            # Compare based on trial equality relation (date, time, hand, AND user_id equal)
            if trials_list[i] in self.trials:
                # get all sessions for all trials
                if how_many_sessions_list is None:
                    for j in range(0,trials_list[i].num_sessions):
                        sessions_list.append(trials_list[i].session_data[j])
                else:
                    # get all sesssions for this trial
                    if how_many_sessions_list[i] is None:
                        for j in range(0,trials_list[i].num_sessions):
                            sessions_list.append(trials_list[i].session_data[j])
                    else:
                        # get sessions up to how_many, starting from the first session 
                        if second_half == False:
                            for j in range(0,how_many_sessions_list[i]):
                                sessions_list.append(trials_list[i].session_data[j])
                        else:
                            # get sessions up to how_many, starting from the last session 
                            for j in range(trials_list[i].num_sessions-how_many_sessions_list[i],trials_list[i].num_sessions):
                                sessions_list.append(trials_list[i].session_data[j])
            else:
                raise ValueError('Trial',i,'not found in self.trials. Missing trial is',trials_list[i])
        return sessions_list
                
    # returns number of sessions in the specified trials
    def getNumSessions(self, trials_list):
        num_sessions_list = np.zeros(len(trials_list))
        for i in range(len(trials_list)):
            # Compare based on trial equality relation (date, time, hand, AND user_id equal)
            if trials_list[i] in self.trials:
                num_sessions_list[i] = trials_list[i].num_sessions
            else:
                raise ValueError('Trial',i,'not found in self.trials. Missing trial is',trials_list[i])
        return num_sessions_list.astype(int)
        
#%% FUNCTION - CREATE TRIAL OBJECT CLASS
# TODO: define data structure that defines what e.g. contact data means for a given trial type
# TODO: write function to check if any timestamp issues in data as we import it (e.g. a big jump in time)
# TODO: add method to auto-check contact arrays for quick pulses / errors during the import
# TODO: write function to remove a particular session and update the num_sessions
# TODO: define a better user prompt for new trials

class Trial(object):
    def __init__(self, filepath):
        
        known_trials_data = dict.fromkeys(['tap', 'guitar_hero_tap_hold', 'passive_motion_using_phone', 'passive_motion_no_task'])
        known_trials_data['tap'] = ['skywalk','accelerometer','gyroscope','magnetometer','quaternion','contact','user_prompt']
        known_trials_data['guitar_hero_tap_hold'] = ['skywalk','skywalk_power','accelerometer','gyroscope','magnetometer','quaternion','contact']
        known_trials_data['passive_motion_using_phone'] = ['skywalk','skywalk_power','accelerometer','gyroscope','magnetometer','quaternion']
        known_trials_data['passive_motion_no_task'] = ['skywalk','skywalk_power','accelerometer','gyroscope','magnetometer','quaternion']

        self.filepath = filepath

        with h5py.File(filepath, "r") as f:
            # Get list of all sessions (note: currently actual session IDs are arbitrary so we relabel as 0, 1, 2, etc.)
            sessions_list = list(f.keys())
            
            ################ TODO remove this - only relevant for old user study data collector before sessions #######################
            if 'skywalk_data' in sessions_list:
                i = 0
                # Init pandas structure for session data
                self.session_data = pd.DataFrame(columns=[i], index=known_trials_data[self.trial_type])

                # Save all session data to the Trial instance
                for data_stream in known_trials_data[self.trial_type]:
                    # Initialize column names for skywalk and IMU data
                    channel_names = None
                    if data_stream == 'skywalk':
                        channel_counter = list(range(np.array(f[(data_stream+'_data')][()]).shape[1]))
                        channel_names = ["CH" + str(x) for x in channel_counter]
                    elif data_stream == 'accelerometer' or data_stream == 'gyroscope' or data_stream == 'magnetometer':
                        channel_counter = ['x','y','z']
                        channel_names = [data_stream + '_' + x for x in channel_counter]
                    elif data_stream == 'quaternion':
                        channel_counter = ['a','b','c','d']
                        channel_names = [data_stream + '_' + x for x in channel_counter]
                    elif data_stream == 'user_prompt':
                        channel_names = ['swipe_direction', 'clicklocx', 'clicklocy', 'mode']

                    # Create the array without timestamps
                    self.session_data[i][data_stream] = pd.DataFrame(np.array(f[(data_stream+'_data')][()]), columns=channel_names)
                    if data_stream == 'contact' or data_stream == 'user_prompt':
                        # Add timestamps in the index
                        self.session_data[i][data_stream].index = np.array(f[('skywalk'+'_timestamps')][()])
                    else:
                        # Add timestamps in the index
                        self.session_data[i][data_stream].index = np.array(f[(data_stream+'_timestamps')][()])
                    
                #######################################################################################################################

            else:
                
                metadata = list(f['metadata'][()])
                # Remove metadata from sessions_list to ensure we don't iterate over it
                sessions_list.remove('metadata')
                sessions_list.remove('__DATA_TYPES__')
                
                # Verify trial type and its constituent data streams are known
                if metadata[0].decode('utf-8') not in known_trials_data:
                    raise ValueError("Specified trial_type not a key in known_trials_data. Specified trial_type is:",metadata[0].decode('utf-8'),". Known trials are:",list(known_trials_data.keys()),". Either change trial_type or add new trial_type and data list to known_trials_data.")
                    
                # Init trial metadata
                self.trial_type, self.user_id, self.firmware_version, self.hand, self.notes = metadata[0].decode('utf-8'), metadata[1].decode('utf-8'), metadata[3].decode('utf-8'), metadata[4].decode('utf-8'), metadata[5].decode('utf-8')
                self.date = datetime.strptime(metadata[2].decode('utf-8'), '%Y-%m-%dT%H-%M-%S').date()
                self.time = datetime.strptime(metadata[2].decode('utf-8'), '%Y-%m-%dT%H-%M-%S').time()
                
                # Init pandas structure for session data
                self.session_data = pd.DataFrame(columns=list(range(len(sessions_list))), index=known_trials_data[self.trial_type])
                self.num_sessions = len(sessions_list)
                
                # Save all session data to the Trial instance
                for i in range(len(sessions_list)):
                    for data_stream in known_trials_data[self.trial_type]:
                        # Initialize column names for skywalk and IMU data
                        channel_names = None
                        if data_stream == 'skywalk':
                            channel_counter = list(range(np.array(f[sessions_list[i]][(data_stream+'_data')][()]).shape[1]))
                            channel_names = ["CH" + str(x) for x in channel_counter]
                        elif data_stream == 'accelerometer' or data_stream == 'gyroscope' or data_stream == 'magnetometer':
                            channel_counter = ['x','y','z']
                            channel_names = [data_stream + '_' + x for x in channel_counter]
                        elif data_stream == 'quaternion':
                            channel_counter = ['a','b','c','d']
                            channel_names = [data_stream + '_' + x for x in channel_counter]
                        elif data_stream == 'user_prompt':
                            channel_names = ['swipe_direction', 'clicklocx', 'clicklocy', 'mode']
    
                        # Create the array without timestamps
                        self.session_data[i][data_stream] = pd.DataFrame(np.array(f[sessions_list[i]][(data_stream+'_data')][()]), columns=channel_names)
                        # Add timestamps in the index
                        self.session_data[i][data_stream].index = np.array(f[sessions_list[i]][(data_stream+'_timestamps')][()])
                        
                        
        
    def __repr__(self):
        return f'Trial <trial_type={self.trial_type}, user_id={self.user_id}, date={self.date}, firmware_version={self.firmware_version}, hand={self.hand}, num_sessions={self.num_sessions}>'
    
    def __eq__(self, other):
        if type(other) is type(self):
            return (self.date == other.date and self.time == other.time and self.user_id == other.user_id and self.hand == other.hand)
        return False
    
#%% FUNCTION - RESAMPLES ONE TIMESTAMPED ARRAY TO THE SHAPE OF ANOTHER
# resampled_accelerometer_data = resampleData(session['accelerometer'], session['skywalk'])
# TODO make this smarter so it just interleaves the timestamps regardless, and can handle offset (e.g. if we truncate the skwyalk data with meanSubtract)
# TODO realized that it may be easiest to resample using the pandas method
def resampleData(data_to_resample, correct_size_data):
    if (np.abs(data_to_resample.index[0] - correct_size_data.index[0]) > 500000) or (np.abs(data_to_resample.index[-1] - correct_size_data.index[-1]) > 500000):
        print("WARNING: array start and/or end indices are over 1/2 second off. data_to_resample indices are", data_to_resample.index, "correct_size_data indices are: ", correct_size_data.index)
    new_timestamps = TimeSeriesResampler(sz=correct_size_data.shape[0]).fit_transform(data_to_resample.index)
    new_timestamps = np.squeeze(new_timestamps)
    resampled_data = pd.DataFrame(index=new_timestamps, columns = data_to_resample.columns)
    
    for col in data_to_resample.columns:
        intermediate_resampled_data = TimeSeriesResampler(sz=correct_size_data.shape[0]).fit_transform(data_to_resample[col].values)
        intermediate_resampled_data = np.squeeze(intermediate_resampled_data)
        resampled_data[col] = intermediate_resampled_data
        
    return resampled_data

# FUNCTION - Resamples all IMU data to the size of Skywalk data for a given list of sessions (NOT in place)
def resampleIMUData(sessions_list):
    new_sessions_list = [None]*len(sessions_list)
    for i in range(len(sessions_list)):
        new_sessions_list[i] = sessions_list[i].copy()
        new_sessions_list[i]['accelerometer'] = resampleData(new_sessions_list[i]['accelerometer'], new_sessions_list[i]['skywalk'])
        new_sessions_list[i]['gyroscope'] = resampleData(new_sessions_list[i]['gyroscope'], new_sessions_list[i]['skywalk'])
        new_sessions_list[i]['magnetometer'] = resampleData(new_sessions_list[i]['magnetometer'], new_sessions_list[i]['skywalk'])
        new_sessions_list[i]['quaternion'] = resampleData(new_sessions_list[i]['quaternion'], new_sessions_list[i]['skywalk'])
    return new_sessions_list

# FNCTION - drops unshared indices between IMU and Skywalk data
# TODO combine this function with the resampleData function so it actually interleaves timestamps in an intelligent way
def correctIMUIndices(sessions_list, mean_width):
    new_sessions_list = [None]*len(sessions_list)
    for i in range(len(sessions_list)):
        new_sessions_list[i] = sessions_list[i].copy()
        new_sessions_list[i]['accelerometer'] = new_sessions_list[i]['accelerometer'].iloc[mean_width:]
        new_sessions_list[i]['gyroscope'] = new_sessions_list[i]['gyroscope'].iloc[mean_width:]
        new_sessions_list[i]['magnetometer'] = new_sessions_list[i]['magnetometer'].iloc[mean_width:]
        new_sessions_list[i]['quaternion'] = new_sessions_list[i]['quaternion'].iloc[mean_width:]

    return new_sessions_list
    

#%% FUNCTION - TAKES IN PREPROCESSED LIST OF SESSIONS, OUTPUTS A TIMESERIES DATASET 
# TODO: Eliminate the timeseries issues from just concatenating the sessions together
# TODO: make this predict the future rather than predicting the middle of the array
from tensorflow.keras.preprocessing import timeseries_dataset_from_array
import tensorflow as tf
from tensorflow import keras
from sklearn import preprocessing

def timeseriesFromSessionsList(sessions_list, sequence_length, fit_scaler = False, scaler_to_use = None, IMU_data = None):
    labels_array = np.empty((0,))
    data_array = np.empty((0,sessions_list[0]['skywalk'].shape[1]))
    for session in sessions_list:
        # Collapse labels onto skywalk timestamps
        labels_array = np.append(labels_array, np.array(session['contact'][0][session['skywalk'].index]), axis=0)
        partial_data_array = np.array(session['skywalk'])                   
        # add IMU data if applicable
        if IMU_data is not None:
            for data_stream in IMU_data:
                partial_data_array = np.append(partial_data_array, np.array(session[data_stream]), axis=1)
        data_array = np.append(data_array, np.array(session['skywalk']), axis=0)

    # if fitting a new scaler
    if fit_scaler:
        if scaler_to_use is not None:
            raise ValueError("Cannot assign scaler and fit a new one! Either change fit_scaler to False or remove scaler_to_use.")
        scaler = preprocessing.StandardScaler().fit(data_array)
        data_array = scaler.transform(data_array)
        out_dataset = tf.keras.preprocessing.timeseries_dataset_from_array(data_array, labels_array, sequence_length=sequence_length, sequence_stride=1, sampling_rate=1, batch_size=32, shuffle=False, seed=None, start_index=None, end_index=None)
        return out_dataset, data_array, labels_array, scaler
    
    # If scaler was provided (e.g. this is test data)
    elif scaler_to_use is not None:
        data_array = scaler_to_use.transform(data_array)
        out_dataset = tf.keras.preprocessing.timeseries_dataset_from_array(data_array, labels_array, sequence_length=sequence_length, sequence_stride=1, sampling_rate=1, batch_size=32, shuffle=False, seed=None, start_index=None, end_index=None)
        return out_dataset, data_array, labels_array
    
    # Default, no scaler at all
    else:
        out_dataset = tf.keras.preprocessing.timeseries_dataset_from_array(data_array, labels_array, sequence_length=sequence_length, sequence_stride=1, sampling_rate=1, batch_size=32, shuffle=False, seed=None, start_index=None, end_index=None)
        return out_dataset, data_array, labels_array

#%% FUNCTION - SCALES SKYWALK DATA BY DIVIDING BY POWER ARRAY ('skywalk_powerscaled')
# TODO: Fix bug - the power can switch after the first half of the channels have gone, but not the second!!!
# TODO: (cont.) there's currently a 50% chance any single sample gets incorrectly scaled
# TODO: fix another bug because the backscatter channels have more power inherently, don't want to scale the neighbors back too far
def powerScaleSkywalkData(sessions_list):
    for session in sessions_list:
        # First generate array equal in length to session['skywalk'] that has the power at each timepoint
        power_copy = session['skywalk_power'].copy()
        # Copy power at Ch16 -> Ch17 and Ch18 -> Ch19
        # Ch17 and Ch19 currently just use 16 and 18 power level in firmware (shared timeslot LED)
        power_copy[17], power_copy[19] = power_copy[16], power_copy[18]
        power_copy2 = power_copy.drop(columns=[0,1,2,3,4,5,6])
        power_copy2.columns += 13
        extended_power_array = pd.concat([power_copy, power_copy2], axis=1)
        temp_power_array = pd.DataFrame(np.zeros(session['skywalk'].shape),index=session['skywalk'].index)
        first_power_update = True
        for ind in extended_power_array.index:
            if first_power_update:
                temp_power_array[temp_power_array.index <= ind] = np.array(extended_power_array.loc[ind])
                first_power_update = False
            temp_power_array[temp_power_array.index > ind] = np.array(extended_power_array.loc[ind])
        temp_power_array.columns = session['skywalk'].columns
        session['skywalk_powerscaled'] = session['skywalk'].div(temp_power_array)

#%% FUNCTION - Take array and subtract the mean of the past n samples from each channel
# Returns a mean-subtracted copy of the array with n fewer rows
def takeMeanDiff(data, n):
  new = data.copy()
  for i in range(n,len(data)):
    new[i] = data[i] - np.mean(data[i-n:i], axis=0)
  return new[n:]

# FUNCTION - does mean-subtraction of skywalk data for all sessions in a list of sessions
def meanSubtractSkywalkData(sessions_list, mean_width, power_scale=False):
    new_sessions_list = [None]*len(sessions_list)
    for i in range(len(sessions_list)):
        new_sessions_list[i] = sessions_list[i].copy()
        if power_scale:
            new_sessions_list[i]['skywalk'] = pd.DataFrame(takeMeanDiff(np.array(sessions_list[i]['skywalk_powerscaled']), mean_width),columns = sessions_list[i]['skywalk_powerscaled'].columns, index = sessions_list[i]['skywalk_powerscaled'].index[mean_width:])
        else:
            new_sessions_list[i]['skywalk'] = pd.DataFrame(takeMeanDiff(np.array(sessions_list[i]['skywalk']), mean_width),columns = sessions_list[i]['skywalk'].columns, index = sessions_list[i]['skywalk'].index[mean_width:])
    return new_sessions_list

#%% FUNCTIONS - v0 CNN model
def applyTimeseriesCNN_v0(trainDataset, epochs, kernel_size, regrate, verbose=1):
    
    for data, labels in trainDataset.take(1):  # only take first element of dataset
      numpy_data = data.numpy()
      numpy_labels = labels.numpy()
    batch_size, sequence_length, n_features = numpy_data.shape[0], numpy_data.shape[1], numpy_data.shape[2]
    
    model = Sequential()
    # 1D convolution across time
    model.add(Conv1D(filters=64, kernel_size=kernel_size, activation='relu',input_shape=(sequence_length,n_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(30, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    # Use categorical crossentropy for one-hot encoded
    # Use sparse categorical crossentropy for 1D integer encoded
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(trainDataset, epochs=epochs, batch_size=batch_size, verbose=verbose)

    # print(model.summary())
    # print(model.evaluate(trainx, trainy))
    
    return model

#%% FUNCTIONS - v1 CNN model
def applyTimeseriesCNN_v1(trainDataset, testDataset, epochs, kernel_size, regrate, verbose=1):
    
    for data, labels in trainDataset.take(1):  # only take first element of dataset
      numpy_data = data.numpy()
      numpy_labels = labels.numpy()
    batch_size, sequence_length, n_features = numpy_data.shape[0], numpy_data.shape[1], numpy_data.shape[2]
    
    model = Sequential()
    # 1D convolution across time
    model.add(Conv1D(filters=64, kernel_size=kernel_size, activation='relu',input_shape=(sequence_length,n_features)))
    model.add(Conv1D(filters=24, kernel_size=kernel_size, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(30, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    # Use categorical crossentropy for one-hot encoded
    # Use sparse categorical crossentropy for 1D integer encoded
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(trainDataset, epochs=epochs, batch_size=batch_size, verbose=verbose)

    # print(model.summary())
    # print(model.evaluate(trainx, trainy))
    
    predictions_test = model.predict(testDataset)
    pred_array = np.array(predictions_test)
    predictions = np.zeros(pred_array.shape[0])
    for i in range(pred_array.shape[0]):
        predictions[i] = np.argmax(pred_array[i,:])
    return model, predictions

# FUNCTION - GET PREDICTIONS
def getPredictions(model, test_dataset):
    predictions_test = model.predict(test_dataset)
    pred_array = np.array(predictions_test)
    predictions = np.zeros(pred_array.shape[0])
    for i in range(pred_array.shape[0]):
        predictions[i] = np.argmax(pred_array[i,:])
    return predictions

#%% FUNCTION - CHECK ACCURACY OF THE MODEL (currently unused, defaulting to applyFlagPulses)
# TODO needs to output RMS and max onset timing error, RMS and max offset timing error
# TODO needs to output # true positives, # false positives, # false negatives, and # false releases (midway through event)
# TODO needs to do this quickly

# def checkAccuracy(model, test_dataset, test_labels_array, sequence_length, correct_tolerance):
#     predictions_test = model.predict(test_dataset)
#     predictions = np.argmax(np.array(predictions_test), axis=1)
#     matched_test_labels_array = test_labels_array[0:predictions.shape[0]]
        
#     # Get actual edges
#     rising_edge_timestamps, rising_edge_indices = getRisingEdgeIndices(matched_test_labels_array)
#     falling_edge_timestamps, falling_edge_indices = getFallingEdgeIndices(matched_test_labels_array)
    
#     correct_array = np.zeros(rising_edge_indices.shape)
    
#     # Get predicted edges
#     pred_rising_edge_timestamps, pred_rising_edge_indices = getRisingEdgeIndices(predictions)
#     pred_falling_edge_timestamps, pred_falling_edge_indices = getFallingEdgeIndices(predictions)

#     # Check if any edges would be combined by the correct tolerance
#     if (max(abs(falling_edge_indices[:falling_edge_indices.shape[0]-1] - rising_edge_indices[1:])) < 2*correct_tolerance):
#         print("One or more events would be combined in this accuracy metric, defaulting to zero tolerance")
#     # Expand events if it won't break stuff
#     else:
#         for ind in rising_edge_indices:
#             if (ind - correct_tolerance) <= 0:
#                 matched_test_labels_array[0:ind] = matched_test_labels_array[ind]
#             else:
#                 matched_test_labels_array[ind-correct_tolerance:ind] = matched_test_labels_array[ind]
        
#         for ind in falling_edge_indices:
#             if (ind + correct_tolerance) >= matched_test_labels_array.shape[0]:
#                 matched_test_labels_array[ind:matched_test_labels_array.shape[0]] = matched_test_labels_array[ind]
#             else:
#                 matched_test_labels_array[ind:ind+correct_tolerance] = matched_test_labels_array[ind]    
                
#     # Multiply matched_test_labels_array by predictions to get overlap
#     mult_array = matched_test_labels_array * predictions
#     mult_rising_edge_timestamps, mult_rising_edge_indices = getRisingEdgeIndices(mult_array)
#     mult_falling_edge_timestamps, mult_falling_edge_indices = getFallingEdgeIndices(mult_array)
    
#     # Number of correct predictions (need to verify 1:1 correspondence between prediction and actual)
#     num_correct = mult_rising_edge_indices.shape[0]
    
#     for i in range(rising_edge_indices.shape[0]):
#         for j in range(pred_rising_edge_indices.shape[0]):
#             if pred_rising_edge_indices[j] - rising_edge_indices
        

#     # for i in range(rising_edge_indices.shape[0]):
#     #     mult_rising_edge_timestamps, mult_rising_edge_indices = getRisingEdgeIndices(mult_array[rising_edge_indices[i]:falling_edge_indices[i]])
#     #     mult_falling_edge_timestamps, mult_falling_edge_indices = getFallingEdgeIndices(mult_array[rising_edge_indices[i]:falling_edge_indices[i]])
#     #     if (mult_falling_edge_indices.shape[0] >= 1):
#     #         if (mult_rising_edge_indices.shape[0] >=1):
#     #             for ind in mult_falling_edge_indices:
#     #                 if ind < 
                    
#     # Add matched_test_labels_array to predictions to check for false releases
#     sum_array = matched_test_labels_array + predictions

    
#     # metrics    
    
#     # call out if num_pred_rising_edges != num_correct_predictions (e.g. two or more predictions got combined)?
#     # can also get handled in the latency calc actually...
    
#     # expand actual events 

#%% Flag Pulses (applyFlagPulses)

def applyFlagPulses(predicted, actual):
    correct=0
    false_pos=0
    false_neg=0
    N=len(predicted)
    pred_on=False
    true_on=False
    caught=False
    caught_vec = []
    true_on_vec = []
    pred_on_vec = []
    
    for i in range(N):
        caught_vec.append(caught)
        true_on_vec.append(true_on)
        pred_on_vec.append(pred_on)
        
        if true_on and pred_on: # both are high
            if actual[i]==0 and predicted[i]==0: 
                true_on=False
                pred_on=False
                correct+=1 # increase number correct
                caught = True # the press has been caught
            elif actual[i]==0:
                true_on=False
                correct+=1 # increase number correct
                caught = True # the press has been caught
            elif predicted[i]==0:
                pred_on=False
                correct+=1 # increase number correct
                caught = True # the press has been caught
                
        elif true_on and not pred_on: 
            if actual[i]==0: # see if we missed a pulse (false negative)
                true_on=False
                if caught == False: # we haven't caught the pulse yet
                    false_neg+=1 # increase false negative if true_on and not pred_on and actual = 0
                    true_on=False
            if predicted[i]==1:
                pred_on=True
        elif pred_on and not true_on:
            if predicted[i]==0: # see if we detected a fake pulse (false positive)
                if caught == False:
                    false_pos+=1 # increase false positive if pred_on and not true_on and actual = 0
                    #print("false positive at:", i)
                pred_on=False
            if actual[i]==1:
                true_on=True
        
        else: # both are low
            caught = False # waiting for the next pulse, so haven't caught it yet
            if actual[i]==1:
                true_on=True
            if predicted[i]==1:
                pred_on=True
        
    return correct,false_pos,false_neg, caught_vec, true_on_vec, pred_on_vec


#%% FUNCTION - GET INDICES OF RISING EDGES FOR DATAFRAME
# contact_dataframe = train_sessions_list[0]['contact']
# which_column = 0
# rising_edge_timestamps, rising_edge_indices = getRisingEdgeIndices(contact_dataframe, which_column)
def getRisingEdgeIndices(input_df, which_column):
    if (type(input_df) == pd.DataFrame):
        saved_indices = input_df.index
        df = input_df.reset_index()[input_df.columns[which_column]]
        
        # Temporary new rows
        new_first_row  = pd.DataFrame([0], index = [0])
        new_last_row  = pd.DataFrame([0], index = [(max(df.index)+1)])
        
        # Right-shift the array (rising), then flip the ones to zeros
        df.index += 1
        rising = (~(pd.concat([new_first_row, df]).astype(bool))).astype(float)
        # Multiply by the original and get the indices of the remaining ones to get rising edges
        df.index -= 1
        rising_edge_df = rising * pd.concat([df,new_last_row])
        rising_edge_indices = rising_edge_df.index[rising_edge_df[0] == True]
        rising_edge_timestamps = saved_indices[rising_edge_indices]
        return rising_edge_timestamps, rising_edge_indices
    # 1D array case
    elif (type(input_df) == np.ndarray):
        df = input_df
        # Right-shift the array (rising), then flip the ones to zeros
        rising = (~(np.insert(df, 0, 0).astype(bool))).astype(float)
        # Multiply by the original and get the indices of the remaining ones to get rising edges
        rising_edge_df = rising * np.append(df,0)
        rising_edge_indices = np.where(rising_edge_df == True)[0]
        return rising_edge_indices

# FUNCTION - GET INDICES OF FALLING EDGES FOR DATAFRAME
# contact_dataframe = train_sessions_list[0]['contact']
# which_column = 0
# falling_edge_timestamps, falling_edge_indices = getFallingEdgeIndices(contact_dataframe, which_column)
def getFallingEdgeIndices(input_df, which_column):
    if (type(input_df) == pd.DataFrame):
        saved_indices = input_df.index
        df = input_df.reset_index()[input_df.columns[which_column]]
        
        # Temporary new rows
        new_first_row  = pd.DataFrame([0], index = [0])
        new_last_row  = pd.DataFrame([0], index = [(max(df.index)+1)])
        
        # Left-shift the array (falling), then flip the ones to zeros
        falling = (~(pd.concat([df, new_last_row]).astype(bool))).astype(float)
        df.index += 1
        # Multiply by the original and get the indices of the remaining ones to get rising edges
        falling_edge_df = falling * pd.concat([new_first_row,df])
        falling_edge_indices = falling_edge_df.index[falling_edge_df[0] == True]
        falling_edge_timestamps = saved_indices[falling_edge_indices]
        return falling_edge_timestamps, falling_edge_indices

    # 1D array case
    elif (type(input_df) == np.ndarray):
        df = input_df[:,which_column]
        
        # Left-shift the array (falling), then flip the ones to zeros
        falling = (~(np.append(df,0).astype(bool))).astype(float)
        # Multiply by the original and get the indices of the remaining ones to get falling edges
        falling_edge_df = falling * np.insert(df, 0, 0)
        falling_edge_indices = np.where(falling_edge_df == True)[0]
        return falling_edge_indices
    
#%% PLOT PREDICTED VS. ACTUAL DATA / LABELS
# plotPredictions(predictions, test_labels_array, test_data_array)
def plotPredictions(predictions, labels_array, data_array):
    pred_rising_edges = getRisingEdgeIndices(predictions, which_column=None)
    pred_falling_edges = getFallingEdgeIndices(predictions, which_column=None)
    for i in range(len(pred_rising_edges)):
        plt.axvline(pred_rising_edges[i], color="red", ymin=.8, ymax=1, alpha=0.1)
        plt.axvspan(pred_rising_edges[i], pred_falling_edges[i], ymin=.8, ymax=1, facecolor='r', alpha=0.3, label =  "_"*i + "prediction")
        plt.axvline(pred_falling_edges[i], color="red", ymin=.8, ymax=1, alpha=0.1)
    
    real_rising_edges = getRisingEdgeIndices(labels_array, which_column=None)
    real_falling_edges = getFallingEdgeIndices(labels_array, which_column=None)
    for i in range(len(real_rising_edges)):
        plt.axvline(real_rising_edges[i], color="blue", ymin=0, ymax=.8, alpha=0.1)
        plt.axvspan(real_rising_edges[i], real_falling_edges[i], ymin=0, ymax=.8, facecolor='b', alpha=0.3, label =  "_"*i + "actual")
        plt.axvline(real_falling_edges[i], color="blue", ymin=0, ymax=.8, alpha=0.1)
    
    plt.plot(data_array)
    plt.legend()
    return

#%% FUNCTION - RANDOMLY CHOOSE N SAMPLES FROM SESSIONS
# test_sessions, train_sessions = sampleNSessions(sessions_list, n_sessions)
def sampleNSessions(sessions_list, n_sessions):
    random.seed()
    temp = random.sample(sessions_list, len(sessions_list))
    return temp[:n_sessions], temp[n_sessions:]


#%% DATA IMPORTING, PROCESSING, AND ML PIPELINE
# 
# 1. Import all data into User and Trial data structure [eventually this will be the structure of the database we fetch from]
# 2. Subselect a list of the trials you want from each user by using the getTrials() method.
# 3. From the trials you've selected, pull out the sessions you want into a train_sessions_list and test_sessions_list using getSessions()
# 4. Run stage 1 preprocessing functions on the sessions_lists before sessions get concatenated - e.g. mean-subtraction, scaling by LED power
# 5. Select data and label columns, concatenate sessions_lists into big train_data/labels and test_data/labels arrays. 
# 6. Generate timeseries dataset from processed train_data/labels and test_data/labels.
# 7. Train and test network, report accuracy, plot predictions
#%% IMPORT DATA
dirpath = '../dataset/tylerchen-guitar-hero-tap-hold/'
allFiles = [f for f in listdir(dirpath) if (isfile(join(dirpath, f)) and f.endswith(".h5"))]
tylerchen = User('tylerchen')
for filepath in allFiles:
    tylerchen.appendTrial(Trial(dirpath+filepath))

# Define subsets of sessions
simple_sessions_list = tylerchen.getSessions(tylerchen.getTrials(date='2022-03-06', trial_type='guitar_hero_tap_hold', notes='consistent hand position, no noise'))
drag_sessions_list = tylerchen.getSessions(tylerchen.getTrials(date='2022-03-06', trial_type='guitar_hero_tap_hold', notes='unidirectional motion of hand during hold events, emulating click + drag'))
rotation_sessions_list = tylerchen.getSessions(tylerchen.getTrials(date='2022-03-06', trial_type='guitar_hero_tap_hold', notes='light rotation of wrist in between events'))
flexion_extension_sessions_list = tylerchen.getSessions(tylerchen.getTrials(date='2022-03-06', trial_type='guitar_hero_tap_hold', notes='light flexion extension of wrist during events, stable otherwise'))
open_close_sessions_list = tylerchen.getSessions(tylerchen.getTrials(date='2022-03-06', trial_type='guitar_hero_tap_hold', notes='mix of open and close other fingers, stable otherwise'))
allmixed_sessions_list = tylerchen.getSessions(tylerchen.getTrials(date='2022-03-06', trial_type='guitar_hero_tap_hold', notes='mix of rotation, flexion, move, open/close other fingers'))
allmixed_background_sessions_list = tylerchen.getSessions(tylerchen.getTrials(date='2022-03-06', trial_type='guitar_hero_tap_hold', notes='guitarhero with rotation, motion, fingers open/close, interspersed with idle motion, picking up objects, resting hand'))
allmixed_background_sessions_day2_list = tylerchen.getSessions(tylerchen.getTrials(date='2022-03-07', trial_type='guitar_hero_tap_hold', notes='guitarhero with rotation, motion, fingers open/close, interspersed with idle motion, picking up objects, resting hand'))
background_sessions_list = tylerchen.getSessions(tylerchen.getTrials(date='2022-03-06', trial_type='passive_motion_no_task'))
background_phone_sessions_list = tylerchen.getSessions(tylerchen.getTrials(trial_type = 'passive_motion_using_phone'))
for i in range(len(background_sessions_list)):
    background_sessions_list[i]['contact'] = pd.DataFrame(np.zeros(background_sessions_list[i]['skywalk'].shape[0]), index = background_sessions_list[i]['skywalk'].index)
for i in range(len(background_phone_sessions_list)):
    background_phone_sessions_list[i]['contact'] = pd.DataFrame(np.zeros(background_phone_sessions_list[i]['skywalk'].shape[0]), index = background_phone_sessions_list[i]['skywalk'].index)

all_day_one_sessions_list =  background_phone_sessions_list+background_sessions_list+allmixed_background_sessions_list+allmixed_sessions_list+open_close_sessions_list+ flexion_extension_sessions_list+rotation_sessions_list+drag_sessions_list+simple_sessions_list


#%% Things to try - 
# expand events
# different training subsets
# adding accel/gyro
# adding gravity direction (that's just quaternion i guess)
# PCA project the dataset into 3d and plot the tap vs. non tap? - might need to fourier transform first or smth mmm
# fourier transform each 11 sample segment - compare non-tap vs. tap?
# Note - power scaling seems to dramatically inflate the importance of the backscatter channels (ofc) cuz their power is lower. fix this by making a correction factor or smth
#%% SELECT SESSIONS AND TRAIN MODEL / PREDICT
means = [128]
num_repeats = 10
model_list = [None]*num_repeats
n_test = 5
for mean in means:
    full_correct, full_false_pos, full_false_neg = np.zeros([n_test, num_repeats]),np.zeros([n_test, num_repeats]),np.zeros([n_test, num_repeats])
    best_correct, best_false_pos, best_false_neg, best_index = [0]*n_test, [1000]*n_test, [1000]*n_test, [0]*n_test
    for count in range(num_repeats):
        
        # Create lists of training and testing sessions by sampling from the sessions lists
        simple_test_sessions, simple_train_sessions = sampleNSessions(simple_sessions_list, 5)
        drag_test_sessions, drag_train_sessions = sampleNSessions(drag_sessions_list, 2)
        rotation_test_sessions, rotation_train_sessions = sampleNSessions(rotation_sessions_list, 2)
        flexion_extension_test_sessions, flexion_extension_train_sessions = sampleNSessions(flexion_extension_sessions_list, 1)
        openclose_test_sessions, openclose_train_sessions = sampleNSessions(open_close_sessions_list, 1)
        
        # Concatenate training sessions together, append test sessions into a metalist to get trial-type-specific metrics
        train_sessions_list = rotation_train_sessions+openclose_train_sessions+drag_test_sessions
        test_sessions_metalist = [simple_test_sessions,drag_test_sessions,rotation_test_sessions,flexion_extension_test_sessions,openclose_test_sessions]
        if n_test != len(test_sessions_metalist): raise ValueError("n_test (", n_test,") must equal length of test_sessions_metalist (", len(test_sessions_metalist),")")
        test_descriptions = ['simple', 'drag', 'rotation', 'flexion','open_close']

        # Shuffle training list
        random.seed(10)
        random.shuffle(train_sessions_list)
        
        # 1. Scale skywalk data by the LED power (adds new column 'skywalk_powerscaled')
        power_scale = False # leave as false until issues are fixed
        if (power_scale):
            powerScaleSkywalkData(train_sessions_list)
            for i in range(len(test_sessions_metalist)): powerScaleSkywalkData(test_sessions_metalist[i])
            
        # 2. Resample IMU data (makes a copy)
        train_sessions_list = resampleIMUData(train_sessions_list)
        for i in range(len(test_sessions_metalist)): test_sessions_metalist[i] = resampleIMUData(test_sessions_metalist[i])
        
        # X. Take derivative of accel_data (if relevant)
            
        # 3. Subtract mean from skywalk data (makes a copy)
        train_sessions_list = meanSubtractSkywalkData(train_sessions_list, mean, power_scale)
        for i in range(len(test_sessions_metalist)): test_sessions_metalist[i] = meanSubtractSkywalkData(test_sessions_metalist[i], mean, power_scale)
        
        # 4. Correct IMU indices after mean subtraction happened
        train_sessions_list = correctIMUIndices(train_sessions_list, mean)
        for i in range(len(test_sessions_metalist)): test_sessions_metalist[i] = correctIMUIndices(test_sessions_metalist[i], mean)

        scaled = False
        sequence_length = 11
        # IMU_data = ['accelerometer', 'gyroscope']
        IMU_data = None
        if not scaled:
            # Convert data into timeseries
            train_dataset, train_data_array, train_labels_array = timeseriesFromSessionsList(train_sessions_list, sequence_length, IMU_data = IMU_data)
            test_dataset, test_data_array, test_labels_array = [None]*n_test, [None]*n_test, [None]*n_test
            for i in range(n_test):
                test_dataset[i], test_data_array[i], test_labels_array[i] = timeseriesFromSessionsList(test_sessions_metalist[i], sequence_length, IMU_data = IMU_data)
        
        else:
            train_dataset, train_data_array, train_labels_array, scaler = timeseriesFromSessionsList(train_sessions_list, sequence_length, fit_scaler = True, IMU_data = IMU_data)
            for i in range(n_test):
                test_dataset[i], test_data_array[i], test_labels_array[i] = timeseriesFromSessionsList(test_sessions_metalist[i], sequence_length, scaler_to_use = scaler, IMU_data = IMU_data)
            
        model_list[count] = applyTimeseriesCNN_v0(train_dataset, epochs=7, kernel_size=5, regrate=0.1, verbose=0)
        
        for j in range(len(test_sessions_metalist)):
            predictions = getPredictions(model_list[count], test_dataset[j])
            correct,false_pos,false_neg, caught_vec, true_on_vec, pred_on_vec = applyFlagPulses(predictions, test_labels_array[j])
            if (correct - false_pos) > (best_correct[j] - best_false_pos[j]):
                best_correct[j], best_false_pos[j], best_false_neg[j], best_index[j] = correct, false_pos, false_neg, count
            full_correct[j][count] = correct
            full_false_pos[j][count] = false_pos
            full_false_neg[j][count] = false_neg
            
    for j in range(n_test):
        print(test_descriptions[j])
        print('MEAN: Correct:',"{:.2%}".format(sum(full_correct[j])/(sum(full_correct[j])+sum(full_false_neg[j]))), 'FalsePos:', "{:.2%}".format(sum(full_false_pos[j])/(sum(full_correct[j])+sum(full_false_neg[j]))), 'FalseNeg:', "{:.2%}".format(sum(full_false_neg[j])/(sum(full_correct[j])+sum(full_false_neg[j]))))
        print('BEST: Index:', best_index[j], ' Correct:',"{:.2%}".format(best_correct[j]/(best_correct[j]+best_false_neg[j])), 'FalsePos:', "{:.2%}".format(best_false_pos[j]/(best_correct[j]+best_false_neg[j])), 'FalseNeg:', "{:.2%}".format(best_false_neg[j]/(best_correct[j]+best_false_neg[j])))

#%% SAVE MODEL AND PLOT RESULTS
j = 4
predictions = getPredictions(model_list[6], test_dataset[j])
plotPredictions(predictions, test_labels_array[j], test_data_array[j])
model_list[6].save("../models/2022_03_14_TimeseriesCNN_HL2_v0")


#%% WITHIN-SESSION MODEL


openclose_test_sessions, openclose_train_sessions = sampleNSessions(open_close_sessions_list, 1)
