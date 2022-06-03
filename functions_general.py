# -*- coding: utf-8 -*-
"""
Created on Tues Mar  8 13:09:52 2022

@author: tyler 
"""

# %% Functions to import from this file
# from functions_general import get_rising_edge_indices
# from functions_general import get_falling_edge_indices
# from functions_general import sample_n_sessions

# %% Top-Level Imports
from typing import Union, List

import pandas as pd
import numpy as np
import random


# %% FUNCTION - GET INDICES OF RISING EDGES FOR DATAFRAME
# contact_dataframe = train_sessions_list[0]['contact']
# which_column = 0
# rising_edge_timestamps, rising_edge_indices = get_rising_edge_indices(contact_dataframe, which_column)
def get_rising_edge_indices(input_df, which_column):
    if type(input_df) == pd.DataFrame:
        saved_indices = input_df.index
        df = input_df.reset_index()[input_df.columns[which_column]]

        # Temporary new rows
        new_first_row = pd.DataFrame([0], index=[0])
        new_last_row = pd.DataFrame([0], index=[(max(df.index) + 1)])

        # Right-shift the array (rising), then flip the ones to zeros
        df.index += 1
        rising = (~(pd.concat([new_first_row, df]).astype(bool))).astype(float)
        # Multiply by the original and get the indices of the remaining ones to get rising edges
        df.index -= 1
        rising_edge_df = rising * pd.concat([df, new_last_row])
        rising_edge_indices = rising_edge_df.index[rising_edge_df[0]]
        rising_edge_timestamps = saved_indices[rising_edge_indices]
        return rising_edge_timestamps, rising_edge_indices
    # 1D array case
    elif type(input_df) == np.ndarray:
        df = input_df
        # Right-shift the array (rising), then flip the ones to zeros
        rising = (~(np.insert(df, 0, 0).astype(bool))).astype(float)
        # Multiply by the original and get the indices of the remaining ones to get rising edges
        rising_edge_df = rising * np.append(df, 0)
        rising_edge_indices = np.where(rising_edge_df)[0]
        return rising_edge_indices
    else:
        raise TypeError("input_df must be of type np.ndarray or pd.DataFrame")


# FUNCTION - GET INDICES OF FALLING EDGES FOR DATAFRAME
# contact_dataframe = train_sessions_list[0]['contact']
# which_column = 0
# falling_edge_timestamps, falling_edge_indices = get_falling_edge_indices(contact_dataframe, which_column)
def get_falling_edge_indices(input_df, which_column):
    if type(input_df) == pd.DataFrame:
        saved_indices = input_df.index
        df = input_df.reset_index()[input_df.columns[which_column]]

        # Temporary new rows
        new_first_row = pd.DataFrame([0], index=[0])
        new_last_row = pd.DataFrame([0], index=[(max(df.index) + 1)])

        # Left-shift the array (falling), then flip the ones to zeros
        falling = (~(pd.concat([df, new_last_row]).astype(bool))).astype(float)
        df.index += 1
        # Multiply by the original and get the indices of the remaining ones to get rising edges
        falling_edge_df = falling * pd.concat([new_first_row, df])
        falling_edge_indices = falling_edge_df.index[falling_edge_df[0]]
        falling_edge_timestamps = saved_indices[falling_edge_indices]
        return falling_edge_timestamps, falling_edge_indices

    # 1D array case
    elif type(input_df) == np.ndarray:
        df = input_df[:, which_column]

        # Left-shift the array (falling), then flip the ones to zeros
        falling = (~(np.append(df, 0).astype(bool))).astype(float)
        # Multiply by the original and get the indices of the remaining ones to get falling edges
        falling_edge_df = falling * np.insert(df, 0, 0)
        falling_edge_indices = np.where(falling_edge_df)[0]
        return falling_edge_indices


# %% FUNCTION - RANDOMLY CHOOSE N SAMPLES FROM SESSIONS
# test_sessions, train_sessions = sample_n_sessions(sessions_list, n_sessions)
def sample_n_sessions(sessions_list, n_sessions):
    # temp = random.sample(sessions_list, len(sessions_list))
    return sessions_list[len(sessions_list) - n_sessions:], sessions_list[len(sessions_list) - n_sessions:]


def sample_percentage_sessions(sessions_list: [pd.DataFrame], percentage: Union[float, List[float]]):
    if isinstance(percentage, float):
        percentage = [percentage]
    all_samples = sum(len(session) for session in sessions_list)
    before_samples = all_samples * (1 - sum(percentage))
    divider = [before_samples]
    accumulated_percentage = 0
    for this_percentage in percentage:
        accumulated_percentage += this_percentage
        divider += [before_samples + all_samples * accumulated_percentage]
    samples = [[]]
    length = 0
    for session in sessions_list:
        while length + len(session) > divider[len(samples) - 1]:
            samples += [[]]
        samples[-1] += [session]
        length += len(session)
    for _ in range(len(percentage) + 1 - len(samples)):
        samples += [[]]
    return samples[1:] + [samples[0]]
