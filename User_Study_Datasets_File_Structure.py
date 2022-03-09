# -*- coding: utf-8 -*-
"""
Created on Tues Mar  8 13:09:52 2022

@author: tyler 
"""

#%% Time-based 1D CNN Modelv2 (1D Convolution Across Time)
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
# cnn model
from keras.models import Sequential
#from keras.layers import LSTM
from keras import layers
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.convolutional import AveragePooling1D
from sklearn.metrics import accuracy_score

#%% FUNCTION - CREATE USER OBJECT CLASS
# TODO: make trial_types a dictionary where keys are trial types and values are number of trials of that type?
# TODO: add methods for removal of a trial and automatic updating of num_trials
# TODO: add getNumSessions() method for getting total number of sessions? not sure if useful
# TODO: add padding function where you take multiple trials or sessions or shuffled events and concatenate them into a big array with padding so timeseries doesn’t get borked by a sudden edge
# TODO: add method and / or class for combining multiple sessions into a training set?

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
            self.trials.append(trial)
            self.num_trials += 1
            if trial.trial_type not in self.trial_types.keys():
                self.trial_types[trial.trial_type] = 1
            else:
                self.trial_types[trial.trial_type] += 1
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

    
        
#%% FUNCTION - CREATE TRIAL OBJECT CLASS
# TODO: define function to get how many skywalk channels in the data
# TODO: define data structure that defines what e.g. contact data means for a given trial type
# TODO: write function to check if any timestamp issues in data as we import it (e.g. a big jump in time)
# TODO: add method to auto-check contact arrays for quick pulses / errors during the import
# TODO: write function to remove a particular session and update the num_sessions

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

#%% FUNCTION - TAKES IN PREPROCESSED LIST OF SESSIONS, OUTPUTS A TIMESERIES DATASET 
# TODO: Eliminate the timeseries issues from just concatenating the sessions together
# TODO: make this predict the future rather than predicting the middle of the array
from tensorflow.keras.preprocessing import timeseries_dataset_from_array
import tensorflow as tf
from tensorflow import keras

def timeseriesFromSessionsList(sessions_list, sequence_length):
    labels_array = np.empty((0,))
    data_array = np.empty((0,sessions_list[0]['skywalk'].shape[1]))
    for session in sessions_list:
        # Collapse labels onto skywalk timestamps
        labels_array = np.append(labels_array, np.array(session['contact'][0][session['skywalk'].index]), axis=0)
        data_array = np.append(data_array, np.array(session['skywalk']), axis=0)

    out_dataset = tf.keras.preprocessing.timeseries_dataset_from_array(data_array, labels_array, sequence_length=11, sequence_stride=1, sampling_rate=1, batch_size=32, shuffle=False, seed=None, start_index=None, end_index=None)
    return out_dataset, data_array, labels_array

#%% FUNCTIONS - PREPROCESS LIST OF SESSIONS

# TODO write function to convert skywalk and skywalk_power into one datastream
# TODO write function to drop the last two rows of contact data
# TODO write function to mean-subtract the skywalk data and reduce the size of the array
# TODO write function to train a standardscaler on all sessions and return it (see below function)
# TODO write function to take that standardscaler and fit all sessions to it

# def powerScaleSkywalkData(sessions_list):
#     # Power is reported for 20 channels, data is 33 (all 20, then the last 13)
#     # TODO currently 17 and 19 are unused (need to map 16-> 17 and 18-> 19), uncommon that they're different though
    
#     for session in sessions_list:
#         power_copy = session['skywalk_power'].copy()
#         power_copy.drop(columns=[0,1,2,3,4,5,6], inplace = True)
#         power_copy.columns += 13
#         extended_power_array = pd.concat([session['skywalk_power'], power_copy], axis=1)
#         temp_power_array = pd.DataFrame(np.zeros(session['skywalk'].shape),index=session['skywalk'].index)
#         for ind in extended_power_array.index:
#             temp_power_array.loc[[ind]] = np.array(extended_power_array.loc[[ind]])
        
        
#         session['skywalk_scaled'] = session['skywalk']
#     labels_array = np.append(labels_array, np.array(session['contact'][0][session['skywalk'].index]), axis=0)
#     data_array = np.append(data_array, np.array(session['skywalk']), axis=0)

#     return out_dataset, data_array, labels_array

def takeMeanDiff(data, n):

  new = data.copy()

  for i in range(n,len(data)):
    new[i] = data[i] - np.mean(data[i-n:i], axis=0)

  return new[n:]

# TODO make this work for the power divided skywalk data instead
def meanSubtractSkywalkData(sessions_list, mean_width):
    for session in sessions_list:
        session['skywalk'] = pd.DataFrame(takeMeanDiff(np.array(session['skywalk']), mean_width),columns = session['skywalk'].columns, index = session['skywalk'].index[mean_width:])
    return

#%% FUNCTIONS - PREPROCESS LIST OF SESSIONS


def applyTimeseriesCNN_test(trainDataset, testDataset, epochs, kernel_size, regrate, verbose=1):
    
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
    model.add(Dense(6, activation='softmax'))

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



#%%
from datetime import datetime
from os import listdir
from os.path import isfile, join

# DATA IMPORTING, PROCESSING, AND ML PIPELINE
# 1. Import all data into User and Trial data structure [eventually this will be the structure of the database we fetch from]
# 2. Subselect a list of the trials you want from each user by using the getTrials() method.
# 3. From the trials you've selected, pull out the sessions you want into a train_sessions_list and test_sessions_list
# 4. Run stage 1 preprocessing functions on the sessions_lists before sessions get concatenated - e.g. mean-subtraction, scaling by LED power
# 5. Select data and label columns, concatenate sessions_lists into big train_data/labels and test_data/labels arrays. Run stage 2 preprocessing - e.g. standardScaler
# 6. Generate timeseries dataset from processed train_data/labels and test_data/labels.
# 7. Train and test network, report accuracy, plot predictions

dirpath = '../dataset/tylerchen-guitar-hero-tap-hold/'
allFiles = [f for f in listdir(dirpath) if (isfile(join(dirpath, f)) and f.endswith(".h5"))]
tylerchen = User('tylerchen')
for filepath in allFiles:
    tylerchen.appendTrial(Trial(dirpath+filepath))
    
trials_list = tylerchen.getTrials(date='2022-03-06', trial_type='guitar_hero_tap_hold')
train_sessions_list = []
test_sessions_list = []
for i in range(0,10):
    train_sessions_list.append(trials_list[0].session_data[i])
    test_sessions_list.append(trials_list[1].session_data[i])
    
# TODO: scale skywalk data by the LED power

# Subtract mean from skywalk data
meanSubtractSkywalkData(train_sessions_list, 128)
meanSubtractSkywalkData(test_sessions_list, 128)

# Get a bunch of data 

train_dataset, train_data_array, train_labels_array = timeseriesFromSessionsList(train_sessions_list, sequence_length=11)
test_dataset, test_data_array, test_labels_array = timeseriesFromSessionsList(test_sessions_list, sequence_length=11)

model, predictions = applyTimeseriesCNN_test(train_dataset, test_dataset, epochs=5, kernel_size=5, regrate=0.1, verbose=1)

#%% PLOT RESULTS

plt.plot(test_labels_array)
plt.plot(predictions, '.', alpha=0.5)
# Postprocessing steps
# pulse_leg = 20
# preds_TFCNN = np.array(applyDebouncer(binary_pred_array, pulse_leg))
# correct, false_pos, false_neg, caught_vec, _, _ = applyFlagPulses(preds_TFCNN, expanded_testy)
# touch_count_array, num_touches = applyTouchCounter(caught_vec)

# plt.plot(train_sessions_list[0]['skywalk'])
# plt.plot(train_sessions_list[0]['contact'][0]*10000)

#%% GAH FINISH THIS LATER
session = trials_list[5].session_data[1]

power_copy = session['skywalk_power'].copy()
power_copy.drop(columns=[0,1,2,3,4,5,6], inplace = True)
power_copy.columns += 13
extended_power_array = pd.concat([session['skywalk_power'], power_copy], axis=1)
temp_power_array = pd.DataFrame(np.zeros(session['skywalk'].shape),index=session['skywalk'].index)
for ind in extended_power_array.index:
    temp_power_array.loc[[ind]] = np.array(extended_power_array.loc[[ind]])
    
temp_power_array[temp_power_array.index > 'ind']


#%%
def takeMeanDiff(data, n):

  new = data.copy()

  for i in range(n,len(data)):
    new[i] = data[i] - np.mean(data[i-n:i], axis=0)

  return new[n:]