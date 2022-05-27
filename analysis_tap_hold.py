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
import numpy as np
# %% Top-Level Imports
import pandas as pd
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
from functions_ml_torch import SkywalkCnnV1
from functions_ml_torch import get_predictions
from functions_ml_torch import timeseries_from_sessions_list
from functions_postprocessing import apply_flag_pulses
from functions_postprocessing import plot_predictions
from functions_preprocessing import mean_subtract_skywalk_data

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
    simple_test_sessions, simple_train_sessions = sample_percentage_sessions(simple_sessions_list, 0.2)
    drag_test_sessions, drag_train_sessions = sample_percentage_sessions(drag_sessions_list, 0.2)
    real_test_sessions, real_train_sessions = sample_percentage_sessions(realistic_sessions_list, 0.2)
    # real_background_test_sessions, real_background_train_sessions = sample_n_sessions(realistic_background_sessions_list, 1)
    rotation_test_sessions, rotation_train_sessions = sample_percentage_sessions(rotation_sessions_list, 0.2)
    flexion_extension_test_sessions, flexion_extension_train_sessions = sample_percentage_sessions(
        flexion_extension_sessions_list, 0.2)
    allmixed_test_sessions, allmixed_train_sessions = sample_percentage_sessions(
        allmixed_sessions_list, 0.2)
    open_close_test_sessions, open_close_train_sessions = sample_percentage_sessions(open_close_sessions_list, 0.2)

    # Concatenate training sessions, append test sessions into a metalist to get trial-type-specific metrics
    # train_sessions_list = simple_train_sessions + drag_train_sessions + open_close_train_sessions + allmixed_train_sessions + flexion_extension_train_sessions + allmixed_background_sessions_list + allmixed_background_sessions_day2_list + background_sessions_list
    # test_sessions_meta_names = ["simple_test_sessions", "drag_test_sessions", "rotation_test_sessions",
    #                             "flexion_extension_test_sessions", "open_close_test_sessions", "allmixed_test_sessions"]
    # test_sessions_metalist = [simple_test_sessions, drag_test_sessions, rotation_test_sessions,
    #                           flexion_extension_test_sessions, open_close_test_sessions, allmixed_test_sessions]
    train_sessions_list = simple_train_sessions + real_train_sessions + open_close_train_sessions + allmixed_train_sessions + flexion_extension_train_sessions
    test_sessions_meta_names = ["real_test_sessions", "simple_test_sessions", "flexion_extension_test_sessions",
                                "open_close_test_sessions", "allmixed_test_sessions"]
    test_sessions_metalist = [real_test_sessions, simple_test_sessions, flexion_extension_test_sessions,
                              open_close_test_sessions, allmixed_test_sessions]
    n_test = len(test_sessions_metalist)
    if n_test != len(test_sessions_metalist):
        raise ValueError("n_test (", n_test, ") must equal length of test_sessions_metalist (",
                         len(test_sessions_metalist), ")")
    test_descriptions = ['simple', 'drag', 'rotation', 'flexion', 'open_close', 'allmixed']

    # Shuffle training list
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

    scaled = True
    sequence_length = 11
    # IMU_data = ['accelerometer', 'gyroscope']
    IMU_data = None
    test_dataset, test_data_array, test_labels_array = [None] * n_test, [None] * n_test, [None] * n_test
    if not scaled:
        # Convert data into timeseries
        train_dataset, train_data_array, train_labels_array = \
            timeseries_from_sessions_list(train_sessions_list, sequence_length, imu_data=IMU_data, shuffle=True)
        for i in range(n_test):
            test_dataset[i], test_data_array[i], test_labels_array[i] = timeseries_from_sessions_list(
                test_sessions_metalist[i], sequence_length, imu_data=IMU_data)

    else:
        # Convert data into timeseries
        train_dataset, train_data_array, train_labels_array, saved_scaler = timeseries_from_sessions_list(
            train_sessions_list, sequence_length, fit_scaler=True, imu_data=IMU_data, shuffle=True)
        for i in range(n_test):
            test_dataset[i], test_data_array[i], test_labels_array[i] = timeseries_from_sessions_list(
                test_sessions_metalist[i], sequence_length, scaler_to_use=saved_scaler, imu_data=IMU_data)

    num_workers = 0
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=num_workers,
                                  pin_memory=True)
    test_dataloader = [DataLoader(dataset, batch_size=128, num_workers=num_workers, pin_memory=True) for dataset in
                       test_dataset]

    kernel_size = 5
    epochs = 10

    data, labels, weights = next(iter(train_dataloader))
    numpy_data = data.numpy()
    numpy_labels = labels.numpy()
    batch_size, seq_length, n_features = numpy_data.shape[0], numpy_data.shape[1], numpy_data.shape[2]
    model = SkywalkCnnV1(kernel_size, n_features, seq_length, test_sessions_meta_names)

    print(summary(model, data.shape[1:]))
    logger = TensorBoardLogger('logs')

    trainer = Trainer(
        accelerator="cpu",
        max_epochs=epochs,
        logger=logger,
        val_check_interval=1.0,
        callbacks=[
            LearningRateMonitor(logging_interval='epoch')
        ]
    )

    # # Run learning rate finder
    # lr_finder = trainer.tuner.lr_find(model, train_dataloaders=train_dataloader)
    #
    # # Results can be found in
    # print(lr_finder.results)
    #
    # # Plot with
    # fig = lr_finder.plot(suggest=True)
    # fig.show()
    #
    # # Pick point based on plot, or get suggestion
    # new_lr = lr_finder.suggestion()
    #
    # # update hparams of the model
    # model.hparams.lr = new_lr
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)

    # %% EXAMINE DATA
    # colorlist = ['b'] * 8 + ['orange'] * 25
    # alphalist = [1] * 8 + [0.1] * 25
    # for i in range(0, stddf.shape[1]):
    #     plt.plot(stddf.iloc[:,i], alpha = alphalist[i])
    # plt.title("StdDev")
    # plt.legend(stddf.columns)

    # session_descriptions = []
    # all_trials = tylerchen.get_trials()
    # for i in range(len(all_trials)):
    #     if all_trials[i].trial_type == 'guitar_hero_tap_hold':
    #         session_descriptions = session_descriptions + [all_trials[i].notes] * all_trials[i].num_sessions
    #     else:
    #         session_descriptions = session_descriptions + [all_trials[i].trial_type] * all_trials[i].num_sessions
    # print(session_descriptions)
    #
    # fig, axs = plt.subplots(3)
    # axs[0].bar(range(20), np.mean(np.array(meandf)[:,0:20], axis=0))
    # axs[0].bar([3,4,5,6,7,9,10,13,15,16,17,18,19],np.mean(np.array(meandf)[:,20:], axis=0).tolist(), alpha = 0.5)
    # axs[0].set_title('Mean')
    # axs[1].bar(range(20), np.mean(np.array(stddf)[:,0:20], axis=0))
    # axs[1].bar([3,4,5,6,7,9,10,13,15,16,17,18,19],np.mean(np.array(stddf)[:,20:], axis=0).tolist(), alpha = 0.5)
    # axs[1].set_title('StdDev')
    # axs[2].bar(range(20), np.mean(np.array(peakdf)[:,0:20], axis=0))
    # axs[2].bar([3,4,5,6,7,9,10,13,15,16,17,18,19],np.mean(np.array(peakdf)[:,20:], axis=0).tolist(), alpha = 0.5)
    # axs[2].set_title('Peakdiff (max - min)')
    # fig.xlabel('channel')

    # %% SAVE MODEL AND PLOT RESULTS
    i = 0
    j = 5
    # predictions = get_predictions(model_list[i], test_dataset[j])
    # plot_predictions(predictions, test_labels_array[j], test_data_array[j])

    predictions = get_predictions(model, train_dataset)
    plot_predictions(predictions, train_labels_array, train_data_array)
    # model_list[0].save("../models/2022_03_17_TimeseriesCNN_HL2_v1")

    # %% Random stuff
    # plt.plot(test_sessions_metalist[0][0]['skywalk'].iloc[:-11])
    # plt.plot(test_sessions_metalist[0][0]['contact'].iloc[11:][0]*15000)
    i = 0
    for batch in train_dataset:
        inputs, targets = batch
        # if i < 10:
        #     i += 1
        #     continue
        if targets[31] != 1:
            continue
        plot_predictions(np.array([0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]), np.array(targets[2:13]),
                         np.array(inputs[2]))
        # fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)
        # ax1.plot(inputs[17])
        # ax2.plot(inputs[18])
        # ax3.plot(inputs[19])
        # ax4.plot(inputs[20])
        # ax5.plot(inputs[21])
        # ax6.plot(inputs[22])
        print(targets)
        break

    # for data, labels in train_dataset.take(1):  # only take first element of dataset
    #     numpy_data = data.numpy()
    #     numpy_labels = labels.numpy()
    # batch_size, sequence_length, n_features = numpy_data.shape[0], numpy_data.shape[1], numpy_data.shape[2]

    # %% Random stuff