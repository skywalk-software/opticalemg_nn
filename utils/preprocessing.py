# -*- coding: utf-8 -*-
"""
Created on Tues Mar  8 13:09:52 2022

@author: tyler 
"""

# %% Functions to import from this file
# from functions_preprocessing import resample_data
# from functions_preprocessing import resample_imu_data
# from functions_preprocessing import correct_imu_indices
# from functions_preprocessing import power_scale_skywalk_data
# from functions_preprocessing import take_mean_diff
# from functions_preprocessing import mean_subtract_skywalk_data

# %% Top-Level Imports
import pandas as pd
import numpy as np
from tslearn.preprocessing import TimeSeriesResampler

# %% FUNCTION - Take array and subtract the mean of the past n samples from each channel
# Returns a mean-subtracted copy of the array with n fewer rows
def take_mean_diff(data, n):
    new = data.copy()
    for i in range(n, len(data)):
        new[i] = data[i] - np.mean(data[i - n:i], axis=0)
    return new[n:]


# FUNCTION - does mean-subtraction of skywalk data for all sessions in a list of sessions
def mean_subtract_skywalk_data(sessions_list, mean_width, powerscale=False):
    new_sessions_list = [None] * len(sessions_list)
    for i in range(len(sessions_list)):
        new_sessions_list[i] = sessions_list[i].copy()
        if powerscale:
            new_sessions_list[i]['skywalk'] = pd.DataFrame(
                take_mean_diff(np.array(sessions_list[i]['skywalk_powerscaled']), mean_width),
                columns=sessions_list[i]['skywalk_powerscaled'].columns,
                index=sessions_list[i]['skywalk_powerscaled'].index[mean_width:])
        else:
            new_sessions_list[i]['skywalk'] = pd.DataFrame(
                take_mean_diff(np.array(sessions_list[i]['skywalk']), mean_width),
                columns=sessions_list[i]['skywalk'].columns, index=sessions_list[i]['skywalk'].index[mean_width:])
    return new_sessions_list

