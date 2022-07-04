# -*- coding: utf-8 -*-
"""
Created on Tues Mar  8 13:09:52 2022

@author: tyler 
"""

import random
import sys
from os import listdir
from os.path import isfile, join

# import matplotlib.pyplot as plt
import copy
from typing import cast

import numpy as np
# %% Top-Level Imports
import pandas as pd
import torch
import tqdm
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, ConcatDataset
from torchsummary import summary
from tslearn.preprocessing import TimeSeriesResampler

from classes import Trial
# %% Local Imports
from classes import User
from functions_general import sample_percentage_sessions
from functions_ml_torch import SkywalkCnnV1, SkywalkDataset
from functions_ml_torch import get_predictions
from functions_ml_torch import timeseries_from_sessions_list
from functions_postprocessing import apply_flag_pulses
from functions_postprocessing import plot_predictions
from functions_preprocessing import mean_subtract_skywalk_data
#%%
if __name__ == '__main__':

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
    gaze_sessions_list = tylerchen.get_sessions(tylerchen.get_trials(date='2022-07-02'))
    simple_sessions_list = tylerchen.get_sessions(
        tylerchen.get_trials(date='2022-03-06', trial_type='guitar_hero_tap_hold',
                             notes='consistent hand position, no noise'))
    drag_sessions_list = tylerchen.get_sessions(
        tylerchen.get_trials(date='2022-03-06', trial_type='guitar_hero_tap_hold',
                             notes='unidirectional motion of hand during hold events, emulating click + drag'))
    realistic_sessions_list = tylerchen.get_sessions(
        tylerchen.get_trials(date='2022-03-18', trial_type='guitar_hero_tap_hold',
                             notes='realistic HL2 use (longer holds) with rest periods'))
    realistic_background_sessions_list = tylerchen.get_sessions(
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
    background_phone_sessions_list = tylerchen.get_sessions(
        tylerchen.get_trials(trial_type='passive_motion_using_phone'))

    jackieyang = User('jackieyang')
    dirpath = '../dataset/jackieyang-guitar-hero-tap-hold/'
    allFiles = [f for f in listdir(dirpath) if (isfile(join(dirpath, f)) and f.endswith(".h5"))]
    for filepath in allFiles:
        jackieyang.append_trial(Trial(dirpath + filepath))
    jackieyang_sessions_list = jackieyang.get_sessions(jackieyang.get_trials())

    tianshili = User('tianshili')
    dirpath = '../dataset/tianshili-guitar-hero-tap-hold/'
    allFiles = [f for f in listdir(dirpath) if (isfile(join(dirpath, f)) and f.endswith(".h5"))]
    for filepath in allFiles:
        temp_trial = Trial(dirpath + filepath)
        temp_trial.user_id = 'tianshili'
        tianshili.append_trial(temp_trial)
    tianshili_sessions_list = tianshili.get_sessions(tianshili.get_trials())

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
    stddf = pd.DataFrame(columns=all_sessions[0]['skywalk'].columns, index=range(len(all_sessions)))
    i = 0
    for session in all_sessions:
        meandf.loc[i] = session['skywalk'].describe().loc['mean']
        stddf.loc[i] = session['skywalk'].describe().loc['std']
        i += 1

    # %% Things to try -
    # expand events
    # different training subsets
    # adding accel/gyro
    # adding gravity direction (that's just quaternion i guess)
    # PCA project the dataset into 3d and plot the tap vs. non tap? - might need to fourier transform first or smth mmm
    # fourier transform each 11 sample segment - compare non-tap vs. tap?
    # Note - power scaling seems to dramatically inflate the importance of the backscatter channels (ofc)
    #        cuz their power is lower. fix this by making a correction factor or smth
    # %% SELECT SESSIONS AND TRAIN MODEL / PREDICT
    mean = 128
    num_repeats = 10
    model_list = [None] * num_repeats

    # Create lists of training and testing sessions by sampling from the sessions lists
    gaze_val_sessions, gaze_test_sessions, gaze_train_sessions = sample_percentage_sessions(gaze_sessions_list, [0.2, 0.2])
    simple_val_sessions, simple_test_sessions, simple_train_sessions = sample_percentage_sessions(simple_sessions_list,
                                                                                                  [0.2, 0.2])
    drag_val_sessions, drag_test_sessions, drag_train_sessions = sample_percentage_sessions(drag_sessions_list,
                                                                                            [0.2, 0.2])
    real_val_sessions, real_test_sessions, real_train_sessions = sample_percentage_sessions(realistic_sessions_list,
                                                                                            [0.2, 0.2])
    # real_background_val_sessions, real_background_test_sessions, real_background_train_sessions = sample_n_sessions(realistic_background_sessions_list, 1)
    rotation_val_sessions, rotation_test_sessions, rotation_train_sessions = sample_percentage_sessions(
        rotation_sessions_list, [0.2, 0.2])
    flexion_extension_val_sessions, flexion_extension_test_sessions, flexion_extension_train_sessions = sample_percentage_sessions(
        flexion_extension_sessions_list, [0.2, 0.2])
    allmixed_val_sessions, allmixed_test_sessions, allmixed_train_sessions = sample_percentage_sessions(
        allmixed_sessions_list, [0.2, 0.2])
    open_close_val_sessions, open_close_test_sessions, open_close_train_sessions = sample_percentage_sessions(
        open_close_sessions_list, [0.2, 0.2])

    jackieyang_val_sessions, jackieyang_test_sessions, jackieyang_train_sessions = sample_percentage_sessions(
        jackieyang_sessions_list, [0.2, 0.2])
    tianshili_val_sessions, tianshili_test_sessions, tianshili_train_sessions = sample_percentage_sessions(
        tianshili_sessions_list,
        [0.2, 0.2])  # tianshi's has 11 sessions total, arbitrarily choosing 5 sessions for test

    # Concatenate training sessions, append test sessions into a metalist to get trial-type-specific metrics
    # train_sessions_list = simple_train_sessions + drag_train_sessions + open_close_train_sessions + allmixed_train_sessions + flexion_extension_train_sessions + allmixed_background_sessions_list + allmixed_background_sessions_day2_list + background_sessions_list
    # test_sessions_meta_names = ["simple_test_sessions", "drag_test_sessions", "rotation_test_sessions",
    #                             "flexion_extension_test_sessions", "open_close_test_sessions", "allmixed_test_sessions"]
    # test_sessions_metalist = [simple_test_sessions, drag_test_sessions, rotation_test_sessions,
    #                           flexion_extension_test_sessions, open_close_test_sessions, allmixed_test_sessions]
    # tyler_train_sessions_list = simple_train_sessions + real_train_sessions + open_close_train_sessions + allmixed_train_sessions + flexion_extension_train_sessions
    # jackie_train_sessions_list = jackieyang_train_sessions
    # tianshi_train_sessions_list = tianshili_train_sessions
    # test_sessions_meta_names = ["real_test_sessions", "simple_test_sessions", "flexion_extension_test_sessions",
    #                             "open_close_test_sessions", "allmixed_test_sessions", "jackieyang_test_sessions",
    #                             "tianshili_test_sessions"]
    # test_sessions_metalist = [real_test_sessions, simple_test_sessions, flexion_extension_test_sessions,
    #                           open_close_test_sessions, allmixed_test_sessions, jackieyang_test_sessions,
    #                           tianshili_test_sessions]
    # val_sessions_meta_names = ["real_val_sessions", "simple_val_sessions", "flexion_extension_val_sessions",
    #                            "open_close_val_sessions", "allmixed_val_sessions", "jackieyang_val_sessions",
    #                            "tianshili_val_sessions"]
    # val_sessions_metalist = [real_val_sessions, simple_val_sessions, flexion_extension_val_sessions,
    #                          open_close_val_sessions, allmixed_val_sessions, jackieyang_val_sessions,
    #                          tianshili_val_sessions]
    train_sessions_list = gaze_train_sessions
    test_sessions_meta_names = ["gaze_test_sessions"]
    test_sessions_metalist = [gaze_test_sessions]
    val_sessions_meta_names = ["gaze_val_sessions"]
    val_sessions_metalist = [gaze_val_sessions]
    n_test = len(test_sessions_metalist)
    n_val = len(val_sessions_metalist)

    # 1. Scale skywalk data by the LED power (adds new column 'skywalk_powerscaled')
    power_scale = False  # leave as false until issues are fixed
    if power_scale:
        power_scale_skywalk_data(train_sessions_list)
        # power_scale_skywalk_data(tyler_train_sessions_list)
        # power_scale_skywalk_data(jackie_train_sessions_list)
        # power_scale_skywalk_data(tianshi_train_sessions_list)
        for i in range(len(test_sessions_metalist)):
            power_scale_skywalk_data(test_sessions_metalist[i])
        for i in range(len(val_sessions_metalist)):
            power_scale_skywalk_data(val_sessions_metalist[i])

    # 2. Resample IMU data (makes a copy)
    train_sessions_list = resample_imu_data(train_sessions_list)
    # tyler_train_sessions_list = resample_imu_data(tyler_train_sessions_list)
    # jackie_train_sessions_list = resample_imu_data(jackie_train_sessions_list)
    # tianshi_train_sessions_list = resample_imu_data(tianshi_train_sessions_list)
    for i in range(len(test_sessions_metalist)):
        test_sessions_metalist[i] = resample_imu_data(test_sessions_metalist[i])
    for i in range(len(val_sessions_metalist)):
        val_sessions_metalist[i] = resample_imu_data(val_sessions_metalist[i])

    # X. Take derivative of accel_data (if relevant)

    # 3. Subtract mean from skywalk data (makes a copy)
    train_sessions_list = mean_subtract_skywalk_data(train_sessions_list, mean, power_scale)
    # tyler_train_sessions_list = mean_subtract_skywalk_data(tyler_train_sessions_list, mean, power_scale)
    # jackie_train_sessions_list = mean_subtract_skywalk_data(jackie_train_sessions_list, mean, power_scale)
    # tianshi_train_sessions_list = mean_subtract_skywalk_data(tianshi_train_sessions_list, mean, power_scale)
    for i in range(len(test_sessions_metalist)):
        test_sessions_metalist[i] = mean_subtract_skywalk_data(test_sessions_metalist[i], mean, power_scale)
    for i in range(len(val_sessions_metalist)):
        val_sessions_metalist[i] = mean_subtract_skywalk_data(val_sessions_metalist[i], mean, power_scale)

    # 4. Correct IMU indices after mean subtraction happened
    train_sessions_list = correct_imu_indices(train_sessions_list, mean)
    # tyler_train_sessions_list = correct_imu_indices(tyler_train_sessions_list, mean)
    # jackie_train_sessions_list = correct_imu_indices(jackie_train_sessions_list, mean)
    # tianshi_train_sessions_list = correct_imu_indices(tianshi_train_sessions_list, mean)
    for i in range(len(test_sessions_metalist)):
        test_sessions_metalist[i] = correct_imu_indices(test_sessions_metalist[i], mean)
    for i in range(len(val_sessions_metalist)):
        val_sessions_metalist[i] = correct_imu_indices(val_sessions_metalist[i], mean)

    # train_sessions_list = tyler_train_sessions_list + jackie_train_sessions_list + tianshi_train_sessions_list
    train_sessions_list = train_sessions_list

    scaled = False
    sequence_length = 243
    # IMU_data = ['accelerometer', 'gyroscope']
    IMU_data = None
    test_dataset, test_data_array, test_labels_array = [None] * n_test, [None] * n_test, [None] * n_test
    val_dataset, val_data_array, val_labels_array = [None] * n_val, [None] * n_val, [None] * n_val

    if not scaled:
        # Convert data into timeseries
        train_dataset, train_data_array, train_labels_array = \
            timeseries_from_sessions_list(train_sessions_list, sequence_length, imu_data=IMU_data, shuffle=True)
        # tyler_train_dataset, tyler_train_data_array, tyler_train_labels_array = \
        #     timeseries_from_sessions_list(tyler_train_sessions_list, sequence_length, imu_data=IMU_data, shuffle=True)
        # tianshi_train_dataset, tianshi_train_data_array, tianshi_train_labels_array = \
        #     timeseries_from_sessions_list(tianshi_train_sessions_list, sequence_length, imu_data=IMU_data, shuffle=True)
        # jackie_train_dataset, jackie_train_data_array, jackie_train_labels_array = \
        #     timeseries_from_sessions_list(jackie_train_sessions_list, sequence_length, imu_data=IMU_data, shuffle=True)
        for i in range(n_test):
            test_dataset[i], test_data_array[i], test_labels_array[i] = timeseries_from_sessions_list(
                test_sessions_metalist[i], sequence_length, imu_data=IMU_data)
        for i in range(n_val):
            val_dataset[i], val_data_array[i], val_labels_array[i] = timeseries_from_sessions_list(
                val_sessions_metalist[i], sequence_length, imu_data=IMU_data)

    else:
        # Convert data into timeseries
        train_dataset, train_data_array, train_labels_array, saved_scaler = \
            timeseries_from_sessions_list(train_sessions_list, sequence_length, fit_scaler=True, imu_data=IMU_data,
                                          shuffle=True)
        # tyler_train_dataset, tyler_train_data_array, tyler_train_labels_array = \
        #     timeseries_from_sessions_list(tyler_train_sessions_list, sequence_length, scaler_to_use=saved_scaler,
        #                                   imu_data=IMU_data, shuffle=True)
        # tianshi_train_dataset, tianshi_train_data_array, tianshi_train_labels_array = \
        #     timeseries_from_sessions_list(tianshi_train_sessions_list, sequence_length, scaler_to_use=saved_scaler,
        #                                   imu_data=IMU_data, shuffle=True)
        # jackie_train_dataset, jackie_train_data_array, jackie_train_labels_array = \
        #     timeseries_from_sessions_list(tianshi_train_sessions_list, sequence_length, scaler_to_use=saved_scaler,
        #                                   imu_data=IMU_data, shuffle=True)
        for i in range(n_test):
            test_dataset[i], test_data_array[i], test_labels_array[i] = timeseries_from_sessions_list(
                test_sessions_metalist[i], sequence_length, scaler_to_use=saved_scaler, imu_data=IMU_data)
        for i in range(n_val):
            val_dataset[i], val_data_array[i], val_labels_array[i] = timeseries_from_sessions_list(
                val_sessions_metalist[i], sequence_length, scaler_to_use=saved_scaler, imu_data=IMU_data)

    num_workers = 0
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=num_workers, pin_memory=True)
    # tyler_train_dataloader = DataLoader(tyler_train_dataset, batch_size=128, shuffle=True, num_workers=num_workers,
    #                                     pin_memory=True)
    # tianshi_train_dataloader = DataLoader(tianshi_train_dataset, batch_size=128, shuffle=True, num_workers=num_workers,
    #                                       pin_memory=True)
    # jackie_train_dataloader = DataLoader(jackie_train_dataset, batch_size=128, shuffle=True, num_workers=num_workers,
    #                                      pin_memory=True)
    test_dataloader = [DataLoader(dataset, batch_size=128, num_workers=num_workers, pin_memory=True) for dataset in
                       test_dataset]
    # tyler_test_dataloader = [DataLoader(dataset, batch_size=128, num_workers=num_workers, pin_memory=True) for dataset
    #                          in
    #                          test_dataset[:-2]]
    # jackie_test_dataloader = [DataLoader(dataset, batch_size=128, num_workers=num_workers, pin_memory=True) for dataset
    #                           in
    #                           [test_dataset[-2]]]
    # tianshi_test_dataloader = [DataLoader(dataset, batch_size=128, num_workers=num_workers, pin_memory=True) for dataset
    #                            in
    #                            [test_dataset[-1]]]
    val_dataloader = [DataLoader(dataset, batch_size=128, num_workers=num_workers, pin_memory=True) for dataset in
                      val_dataset]
    # tyler_val_dataloader = [DataLoader(dataset, batch_size=128, num_workers=num_workers, pin_memory=True) for dataset in
    #                         val_dataset[:-2]]
    # jackie_val_dataloader = [DataLoader(dataset, batch_size=128, num_workers=num_workers, pin_memory=True) for dataset
    #                          in
    #                          [val_dataset[-2]]]
    # tianshi_val_dataloader = [DataLoader(dataset, batch_size=128, num_workers=num_workers, pin_memory=True) for dataset
    #                           in
    #                           [val_dataset[-1]]]

    kernel_size = 3
    epochs = 5
    next_epochs = 20

    data, labels, weights = next(iter(train_dataloader))
    numpy_data = data.cpu().numpy()
    numpy_labels = labels.cpu().numpy()
    batch_size, seq_length, n_features = numpy_data.shape[0], numpy_data.shape[1], numpy_data.shape[2]
    model = SkywalkCnnV1(kernel_size, n_features, seq_length, val_sessions_meta_names, test_sessions_meta_names)

    print(summary(model, data.shape[1:], device='cpu'))

    CKPT_PATH = "./tylerchen_gaze_20220704.ckpt"

    # %% training
    logger = TensorBoardLogger('logs')

    trainer = Trainer(
        accelerator="cpu" if sys.platform == 'darwin' else "auto",  # temp fix for mps not working
        max_epochs=epochs,
        logger=logger,
        val_check_interval=1.0,
        callbacks=[
            LearningRateMonitor(logging_interval='epoch')
        ]
    )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    # Save a model at CKPT_PATH
    trainer.save_checkpoint(CKPT_PATH)

    # %% Test the model metrics
    trainer = Trainer(
        resume_from_checkpoint=CKPT_PATH
    )
    model.load_from_checkpoint(CKPT_PATH)
    print("test result:")
    print(trainer.validate(model, dataloaders=test_dataloader))

    # %% predict using the first dataset in tyler's val set
    trainer = Trainer(
        resume_from_checkpoint=CKPT_PATH
    )
    model.load_from_checkpoint(CKPT_PATH)
    dataloader = val_dataloader[0]
    dataloader_name = val_sessions_meta_names[0]
    print(f"plotting on {dataloader_name}")
    # device = torch.device("cpu" if sys.platform == 'darwin' else "cuda")
    device = torch.device("cpu")

    model_device = model.to(device)
    
    y_all = []
    y_hat_all = []
    model_device.eval()
    for x, y, w in tqdm.tqdm(dataloader):
        y_hat_float = model_device(x.to(device)).detach().cpu()
        y_hat = (y_hat_float[:, 0] < y_hat_float[:, 1]).long()
        y_hat_all += [y_hat]
        y_all += [y]

    y_all_np = torch.cat(y_all).numpy()
    y_hat_all_np = torch.cat(y_hat_all).numpy()
    # dirty hack to retrieve data from dataloader
    x_all_np = cast(SkywalkDataset, dataloader.dataset).data_array[:len(y_all_np)].numpy()

    fig = plot_predictions(y_hat_all_np, y_all_np, x_all_np)
    fig.show()

    # # For future multi user training

    # model_tyler = copy.deepcopy(model)
    # model_tyler.val_dataset_names = model_tyler.val_dataset_names[: -2]
    # model_tyler.session_type = "tyler"
    # trainer_tyler = Trainer(resume_from_checkpoint=CKPT_PATH, max_epochs=next_epochs, logger=TensorBoardLogger('logs'))
    # trainer_tyler.fit(model_tyler, train_dataloaders=tyler_train_dataloader, val_dataloaders=tyler_val_dataloader)
    #
    # model_jackie = copy.deepcopy(model)
    # model_jackie.val_dataset_names = [model_jackie.val_dataset_names[-2]]
    # model_jackie.session_type = "jackie"
    # trainer_jackie = Trainer(resume_from_checkpoint=CKPT_PATH, max_epochs=next_epochs, logger=TensorBoardLogger('logs'))
    # trainer_jackie.fit(model_jackie, train_dataloaders=jackie_train_dataloader, val_dataloaders=jackie_val_dataloader)
    #
    # model_tianshi = copy.deepcopy(model)
    # model_tianshi.val_dataset_names = [model_tianshi.val_dataset_names[-1]]
    # model_tianshi.session_type = "tianshi"
    # trainer_tianshi = Trainer(resume_from_checkpoint=CKPT_PATH, max_epochs=next_epochs, logger=TensorBoardLogger('logs'))
    # trainer_tianshi.fit(model_tianshi, train_dataloaders=tianshi_train_dataloader, val_dataloaders=tianshi_val_dataloader)
