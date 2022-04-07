# -*- coding: utf-8 -*-
"""
Created on Tues Mar  8 13:09:52 2022

@author: tyler 
"""

# %% Top-Level Imports
import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
import random

from datetime import datetime
from os import listdir
from os.path import isfile, join
from tslearn.preprocessing import TimeSeriesResampler

# %% Local Imports
from classes import User
from classes import Trial

from functions_general import get_rising_edge_indices
from functions_general import get_falling_edge_indices
from functions_general import sample_n_sessions

from functions_preprocessing import take_mean_diff
from functions_preprocessing import mean_subtract_skywalk_data

from functions_ml import timeseries_from_sessions_list
from functions_ml import apply_timeseries_cnn_v0
from functions_ml import apply_timeseries_cnn_v1
from functions_ml import get_predictions

from functions_postprocessing import plot_predictions
from functions_postprocessing import apply_flag_pulses


# %% FUNCTION - RESAMPLES ONE TIMESTAMPED ARRAY TO THE SHAPE OF ANOTHER
# resampled_accelerometer_data = resample_data(session['accelerometer'], session['skywalk'])
# TODO make this smarter so it just interleaves the timestamps regardless, and can handle offset
#  (e.g. if we truncate the skywalk data with meanSubtract)
# TODO realized that it may be easiest to resample using the pandas method
def resample_data(data_to_resample, correct_size_data):
    if (np.abs(data_to_resample.index[0] - correct_size_data.index[0]) > 500000) or (
            np.abs(data_to_resample.index[-1] - correct_size_data.index[-1]) > 500000):
        print("WARNING: array start and/or end indices are over 1/2 second off. data_to_resample indices are",
              data_to_resample.index, "correct_size_data indices are: ", correct_size_data.index)
    new_timestamps = TimeSeriesResampler(sz=correct_size_data.shape[0]).fit_transform(data_to_resample.index)
    new_timestamps = np.squeeze(new_timestamps)
    resampled_data = pd.DataFrame(index=new_timestamps, columns=data_to_resample.columns)

    for col in data_to_resample.columns:
        intermediate_resampled_data = TimeSeriesResampler(sz=correct_size_data.shape[0]).fit_transform(
            data_to_resample[col].values)
        intermediate_resampled_data = np.squeeze(intermediate_resampled_data)
        resampled_data[col] = intermediate_resampled_data

    return resampled_data


# FUNCTION - Resamples all IMU data to the size of Skywalk data for a given list of sessions (NOT in place)
def resample_imu_data(sessions_list):
    new_sessions_list = [None] * len(sessions_list)
    for i_s in range(len(sessions_list)):
        new_sessions_list[i_s] = sessions_list[i_s].copy()
        new_sessions_list[i_s]['accelerometer'] = resample_data(new_sessions_list[i_s]['accelerometer'],
                                                                new_sessions_list[i_s]['skywalk'])
        new_sessions_list[i_s]['gyroscope'] = resample_data(new_sessions_list[i_s]['gyroscope'],
                                                            new_sessions_list[i_s]['skywalk'])
        new_sessions_list[i_s]['magnetometer'] = resample_data(new_sessions_list[i_s]['magnetometer'],
                                                               new_sessions_list[i_s]['skywalk'])
        new_sessions_list[i_s]['quaternion'] = resample_data(new_sessions_list[i_s]['quaternion'],
                                                             new_sessions_list[i_s]['skywalk'])
    return new_sessions_list


# FNCTION - drops unshared indices between IMU and Skywalk data
# TODO combine this function with the resample_data function so it actually interleaves timestamps in an intelligent way
def correct_imu_indices(sessions_list, mean_width):
    new_sessions_list = [None] * len(sessions_list)
    for i_s in range(len(sessions_list)):
        new_sessions_list[i_s] = sessions_list[i_s].copy()
        new_sessions_list[i_s]['accelerometer'] = new_sessions_list[i_s]['accelerometer'].iloc[mean_width:]
        new_sessions_list[i_s]['gyroscope'] = new_sessions_list[i_s]['gyroscope'].iloc[mean_width:]
        new_sessions_list[i_s]['magnetometer'] = new_sessions_list[i_s]['magnetometer'].iloc[mean_width:]
        new_sessions_list[i_s]['quaternion'] = new_sessions_list[i_s]['quaternion'].iloc[mean_width:]

    return new_sessions_list


# %% FUNCTION - SCALES SKYWALK DATA BY DIVIDING BY POWER ARRAY ('skywalk_powerscaled')
# TODO: Fix bug - the power can switch after the first half of the channels have gone, but not the second!!!
# TODO: (cont.) there's currently a 50% chance any single sample gets incorrectly scaled
# TODO: fix another bug because the backscatter channels have more power inherently,
#       don't want to scale the neighbors back too far
def power_scale_skywalk_data(sessions_list):
    for session in sessions_list:
        # First generate array equal in length to session['skywalk'] that has the power at each timepoint
        power_copy = session['skywalk_power'].copy()
        # Copy power at Ch16 -> Ch17 and Ch18 -> Ch19
        # Ch17 and Ch19 currently just use 16 and 18 power level in firmware (shared timeslot LED)
        power_copy[17], power_copy[19] = power_copy[16], power_copy[18]
        power_copy2 = power_copy.drop(columns=[0, 1, 2, 3, 4, 5, 6])
        power_copy2.columns += 13
        extended_power_array = pd.concat([power_copy, power_copy2], axis=1)
        temp_power_array = pd.DataFrame(np.zeros(session['skywalk'].shape), index=session['skywalk'].index)
        first_power_update = True
        for ind in extended_power_array.index:
            if first_power_update:
                temp_power_array[temp_power_array.index <= ind] = np.array(extended_power_array.loc[ind])
                first_power_update = False
            temp_power_array[temp_power_array.index > ind] = np.array(extended_power_array.loc[ind])
        temp_power_array.columns = session['skywalk'].columns
        session['skywalk_powerscaled'] = session['skywalk'].div(temp_power_array)


# %% DATA IMPORTING, PROCESSING, AND ML PIPELINE
# 
# 1. Import all data into User and Trial data structure [eventually will be the structure of the database we fetch from]
# 2. Subselect a list of the trials you want from each user by using the getTrials() method.
# 3. From the trials you've selected, pull out the sessions you want into a
#    train_sessions_list and test_sessions_list using getSessions()
# 4. Run stage 1 preprocessing functions on the sessions_lists before sessions get concatenated -
#    e.g. mean-subtraction, scaling by LED power
# 5. Select data and label columns, concatenate sessions_lists into big train_data/labels and test_data/labels arrays. 
# 6. Generate timeseries dataset from processed train_data/labels and test_data/labels.
# 7. Train and test network, report accuracy, plot predictions
# %% IMPORT DATA
dirpath = '../dataset/tylerchen-guitar-hero-tap-hold/'
allFiles = [f for f in listdir(dirpath) if (isfile(join(dirpath, f)) and f.endswith(".h5"))]
tylerchen = User('tylerchen')
for filepath in allFiles:
    tylerchen.append_trial(Trial(dirpath + filepath))

# Define subsets of sessions
simple_sessions_list = tylerchen.get_sessions(
    tylerchen.get_trials(date='2022-03-06', trial_type='guitar_hero_tap_hold',
                         notes='consistent hand position, no noise'))
drag_sessions_list = tylerchen.get_sessions(
    tylerchen.get_trials(date='2022-03-06', trial_type='guitar_hero_tap_hold',
                         notes='unidirectional motion of hand during hold events, emulating click + drag'))
realistic_sessions_list = tylerchen.get_sessions(
    tylerchen.get_trials(date='2022-03-18', trial_type='guitar_hero_tap_hold',
                         notes='realistic HL2 use (longer holds) with rest periods'))
realistic_background_sessions_list =  tylerchen.get_sessions(
    tylerchen.get_trials(date='2022-03-18', trial_type='passive_motion_no_task'))
    
rotation_sessions_list = tylerchen.get_sessions(
    tylerchen.get_trials(date='2022-03-06', trial_type='guitar_hero_tap_hold',
                         notes='light rotation of wrist in between events'))
flexion_extension_sessions_list = tylerchen.get_sessions(
    tylerchen.get_trials(date='2022-03-06', trial_type='guitar_hero_tap_hold',
                         notes='light flexion extension of wrist during events, stable otherwise'))
open_close_sessions_list = tylerchen.get_sessions(
    tylerchen.get_trials(date='2022-03-06', trial_type='guitar_hero_tap_hold',
                         notes='mix of open and close other fingers, stable otherwise'))
allmixed_sessions_list = tylerchen.get_sessions(
    tylerchen.get_trials(date='2022-03-06', trial_type='guitar_hero_tap_hold',
                         notes='mix of rotation, flexion, move, open/close other fingers'))
allmixed_background_sessions_list = tylerchen.get_sessions(
    tylerchen.get_trials(date='2022-03-06', trial_type='guitar_hero_tap_hold',
                         notes='guitarhero with rotation, motion, fingers open/close, interspersed with '
                               'idle motion, picking up objects, resting hand'))
allmixed_background_sessions_day2_list = tylerchen.get_sessions(
    tylerchen.get_trials(date='2022-03-07', trial_type='guitar_hero_tap_hold',
                         notes='guitarhero with rotation, motion, fingers open/close, interspersed with '
                               'idle motion, picking up objects, resting hand'))
background_sessions_list = tylerchen.get_sessions(
    tylerchen.get_trials(date='2022-03-06', trial_type='passive_motion_no_task'))
background_phone_sessions_list = tylerchen.get_sessions(tylerchen.get_trials(trial_type='passive_motion_using_phone'))
for i in range(len(background_sessions_list)):
    background_sessions_list[i]['contact'] = pd.DataFrame(np.zeros(background_sessions_list[i]['skywalk'].shape[0]),
                                                          index=background_sessions_list[i]['skywalk'].index)
for i in range(len(background_phone_sessions_list)):
    background_phone_sessions_list[i]['contact'] = pd.DataFrame(
        np.zeros(background_phone_sessions_list[i]['skywalk'].shape[0]),
        index=background_phone_sessions_list[i]['skywalk'].index)

all_day_one_sessions_list = background_phone_sessions_list + background_sessions_list + \
                            allmixed_background_sessions_list + allmixed_sessions_list + open_close_sessions_list + \
                            flexion_extension_sessions_list + rotation_sessions_list + drag_sessions_list + \
                            simple_sessions_list

# Collect mean and stddev of all sessions           
all_sessions = tylerchen.get_sessions(tylerchen.get_trials())

meandf = pd.DataFrame(columns=all_sessions[0]['skywalk'].columns, index=range(len(all_sessions)))
stddf =  pd.DataFrame(columns=all_sessions[0]['skywalk'].columns, index=range(len(all_sessions)))
i = 0
for session in all_sessions:
    meandf.loc[i] = session['skywalk'].describe().loc['mean']
    stddf.loc[i] =  session['skywalk'].describe().loc['std']
    i += 1
    
# %% Things to try -
# expand width of tap events (widen allowable detection band)
# different training subsets
# adding accel/gyro
# adding gravity direction (that's just quaternion i guess)
# PCA project the dataset into 3d and plot the tap vs. non tap? - might need to fourier transform first or smth mmm
# fourier transform each 11 sample segment - compare non-tap vs. tap?
# Note - power scaling seems to dramatically inflate the importance of the backscatter channels (ofc)
#        cuz their power is lower. fix this by making a correction factor or smth
# %% SELECT SESSIONS AND TRAIN MODEL / PREDICT
means = [128]
num_repeats = 1
model_list = [None] * num_repeats
n_test = 7
for mean in means:
    full_correct, full_false_pos, full_false_neg = np.zeros([n_test, num_repeats]), np.zeros(
        [n_test, num_repeats]), np.zeros([n_test, num_repeats])
    best_correct, best_false_pos, best_false_neg, best_index = [0] * n_test, [1000] * n_test, [1000] * n_test, \
                                                               [0] * n_test
                                    
    for count in range(num_repeats):
        # Create lists of training and testing sessions by sampling from the sessions lists
        simple_test_sessions, simple_train_sessions = sample_n_sessions(simple_sessions_list, 5)
        drag_test_sessions, drag_train_sessions = sample_n_sessions(drag_sessions_list, 2)
        real_test_sessions, real_train_sessions = sample_n_sessions(realistic_sessions_list, 1)
        # real_background_test_sessions, real_background_train_sessions = sample_n_sessions(realistic_background_sessions_list, 1)
        
        rotation_test_sessions, rotation_train_sessions = sample_n_sessions(rotation_sessions_list, 2)
        flexion_extension_test_sessions, flexion_extension_train_sessions = sample_n_sessions(
            flexion_extension_sessions_list, 1)
        allmixed_test_sessions, allmixed_train_sessions = sample_n_sessions(
            allmixed_sessions_list, 1)
        open_close_test_sessions, open_close_train_sessions = sample_n_sessions(open_close_sessions_list, 1)

        # Concatenate training sessions, append test sessions into a metalist to get trial-type-specific metrics
        # train_sessions_list = simple_train_sessions + real_train_sessions + drag_train_sessions + open_close_train_sessions + allmixed_train_sessions + flexion_extension_train_sessions + allmixed_background_sessions_list + allmixed_background_sessions_day2_list #+ background_sessions_list
        train_sessions_list = simple_train_sessions + real_train_sessions + open_close_train_sessions + allmixed_train_sessions + flexion_extension_train_sessions + allmixed_background_sessions_list + allmixed_background_sessions_day2_list + background_sessions_list

        test_sessions_metalist = [real_test_sessions, simple_test_sessions, drag_test_sessions, rotation_test_sessions,
                                  flexion_extension_test_sessions, open_close_test_sessions, allmixed_test_sessions]
        if n_test != len(test_sessions_metalist):
            raise ValueError("n_test (", n_test, ") must equal length of test_sessions_metalist (",
                             len(test_sessions_metalist), ")")
        test_descriptions = ['real_test', 'simple', 'drag', 'rotation', 'flexion', 'open_close', 'allmixed']

        # Shuffle training list
        random.seed(10)
        random.shuffle(train_sessions_list)

        # 1. Scale skywalk data by the LED power (adds new column 'skywalk_powerscaled')
        power_scale = False  # leave as false until issues are fixed
        if power_scale:
            power_scale_skywalk_data(train_sessions_list)
            for i in range(len(test_sessions_metalist)):
                power_scale_skywalk_data(test_sessions_metalist[i])

        # 2. Resample IMU data (makes a copy)
        train_sessions_list = resample_imu_data(train_sessions_list)
        for i in range(len(test_sessions_metalist)):
            test_sessions_metalist[i] = resample_imu_data(test_sessions_metalist[i])

        # X. Take derivative of accel_data (if relevant)

        # 3. Subtract mean from skywalk data (makes a copy)
        train_sessions_list = mean_subtract_skywalk_data(train_sessions_list, mean, power_scale)
        for i in range(len(test_sessions_metalist)):
            test_sessions_metalist[i] = mean_subtract_skywalk_data(test_sessions_metalist[i], mean, power_scale)

        # 4. Correct IMU indices after mean subtraction happened
        train_sessions_list = correct_imu_indices(train_sessions_list, mean)
        for i in range(len(test_sessions_metalist)):
            test_sessions_metalist[i] = correct_imu_indices(test_sessions_metalist[i], mean)

        # 5. Choose whether we use standardScaler, sequence length, and IMU data
        scaled = True
        sequence_length = 11
        # IMU_data = ['accelerometer', 'gyroscope']
        IMU_data = None
        
        # 6. Convert data into timeseries
        test_dataset, test_data_array, test_labels_array = [None] * n_test, [None] * n_test, [None] * n_test
        if not scaled:
            train_dataset, train_data_array, train_labels_array = \
                timeseries_from_sessions_list(train_sessions_list, sequence_length, imu_data=IMU_data)
            for i in range(n_test):
                test_dataset[i], test_data_array[i], test_labels_array[i] = timeseries_from_sessions_list(
                    test_sessions_metalist[i], sequence_length, imu_data=IMU_data)
        else:
            train_dataset, train_data_array, train_labels_array, saved_scaler = timeseries_from_sessions_list(
                train_sessions_list, sequence_length, fit_scaler=True, imu_data=IMU_data)
            for i in range(n_test):
                test_dataset[i], test_data_array[i], test_labels_array[i] = timeseries_from_sessions_list(
                    test_sessions_metalist[i], sequence_length, scaler_to_use=saved_scaler, imu_data=IMU_data)

        # 7. Train and save the CNN model
        model_list[count] = apply_timeseries_cnn_v1(train_dataset, epochs=10, kernel_size=5, verbose=1)

        # 8. Calculate false neg and false pos for each test condition in the metalist
        for j in range(len(test_sessions_metalist)):
            predictions = get_predictions(model_list[count], test_dataset[j])
            correct, false_pos, false_neg, caught_vec, true_on_vec, pred_on_vec = \
                apply_flag_pulses(predictions, test_labels_array[j])
            if (correct - false_pos) > (best_correct[j] - best_false_pos[j]):
                best_correct[j], best_false_pos[j], best_false_neg[j], best_index[j] = \
                    correct, false_pos, false_neg, count
            full_correct[j][count] = correct
            full_false_pos[j][count] = false_pos
            full_false_neg[j][count] = false_neg

            print(test_descriptions[j])
            print('MEAN: Correct:',
                  "{:.2%}".format(correct / (correct + false_neg)),
                  'FalsePos:',
                  "{:.2%}".format(false_pos / (correct + false_neg)),
                  'FalseNeg:',
                  "{:.2%}".format(false_neg / (correct + false_neg)))

    for j in range(n_test):
        print(test_descriptions[j])
        print('MEAN: Correct:', "{:.2%}".format(sum(full_correct[j]) / (sum(full_correct[j]) + sum(full_false_neg[j]))),
              'FalsePos:', "{:.2%}".format(sum(full_false_pos[j]) / (sum(full_correct[j]) + sum(full_false_neg[j]))),
              'FalseNeg:', "{:.2%}".format(sum(full_false_neg[j]) / (sum(full_correct[j]) + sum(full_false_neg[j]))))
        print('BEST: Index:', best_index[j], ' Correct:',
              "{:.2%}".format(best_correct[j] / (best_correct[j] + best_false_neg[j])), 'FalsePos:',
              "{:.2%}".format(best_false_pos[j] / (best_correct[j] + best_false_neg[j])), 'FalseNeg:',
              "{:.2%}".format(best_false_neg[j] / (best_correct[j] + best_false_neg[j])))


# %% SAVE MODEL AND PLOT RESULTS
i = 0
j = 0
# Plot Model i output on Test Dataset j
# predictions = get_predictions(model_list[i], test_dataset[j])
# plot_predictions(predictions, test_labels_array[j], test_data_array[j])

# Plot Model i output on the whole Training Set
predictions = get_predictions(model_list[i], train_dataset)
plot_predictions(predictions, train_labels_array, train_data_array)
# model_list[0].save("../models/2022_03_17_TimeseriesCNN_HL2_v1")

