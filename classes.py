# -*- coding: utf-8 -*-
"""
Created on Tues Mar  15 10:34:30 2022

@author: tyler 
"""

# %% Imports
import pandas as pd
import numpy as np
import h5py
from datetime import datetime


# %% CLASS - USER
# TODO: add methods for removal of a trial and automatic updating of num_trials
# TODO: add getNumSessions() method for getting total number of sessions? not sure if useful
# TODO: add padding function where you take multiple trials or sessions or shuffled events and
#       concatenate them into a big array with padding so timeseries isn't screwed up by a sudden edge

class User(object):
    def __init__(self, user_id,
                 bio_data=None):
        if bio_data is None:
            bio_data = dict.fromkeys(['name', 'birth_date', 'wrist_circumference', 'race_ethnicity', 'hairy_arms'])
        self.user_id = user_id
        self.trials = []
        self.bio_data = bio_data
        self.trial_types = {}
        self.num_trials = 0

    def __repr__(self):
        return f'User(user_id="{self.user_id}") <bio_data={self.bio_data}, trial_types={self.trial_types}, ' \
               f'num_trials={self.num_trials}>'

    def append_trial(self, trial):
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
            raise ValueError("Trial and User user_id are not matched. Trial user_id:", trial.user_id, "User user_id:",
                             self.user_id)

    # Clear all trials for this user
    def clear_trials(self):
        self.trials = []
        self.num_trials = 0
        self.trial_types = []

    # Set biographical data for this user
    def set_bio_data(self, name=None, birth_date=None, wrist_circumference=None, race_ethnicity=None, hairy_arms=None):
        if name is not None:
            self.bio_data['name'] = name
        if birth_date is not None:
            self.bio_data['birth_date'] = birth_date
        if wrist_circumference is not None:
            self.bio_data['wrist_circumference'] = wrist_circumference
        if race_ethnicity is not None:
            self.bio_data['race_ethnicity'] = race_ethnicity
        if hairy_arms is not None:
            self.bio_data['hairy_arms'] = hairy_arms

    # Clear all biographical data for this user
    def clear_bio_data(self):
        self.bio_data = dict.fromkeys(['name', 'birth_date', 'wrist_circumference', 'race_ethnicity', 'hairy_arms'])

    # Returns a list with all trials that satisfy the search criteria.
    # Adding no criteria is the same as calling user.trials
    def get_trials(self, trial_type=None, date=None, firmware_version=None, hand=None, notes=None):
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
    def get_sessions(self, trials_list, how_many_sessions_list=None, second_half=False):
        sessions_list = []
        for i_t in range(len(trials_list)):
            # Compare based on trial equality relation (date, time, hand, AND user_id equal)
            if trials_list[i_t] in self.trials:
                # get all sessions for all trials
                if how_many_sessions_list is None:
                    for j_s in range(0, trials_list[i_t].num_sessions):
                        sessions_list.append(trials_list[i_t].session_data[j_s])
                else:
                    # get all sessions for this trial
                    if how_many_sessions_list[i_t] is None:
                        for j_s in range(0, trials_list[i_t].num_sessions):
                            sessions_list.append(trials_list[i_t].session_data[j_s])
                    else:
                        # get sessions up to how_many, starting from the first session 
                        if not second_half:
                            for j_s in range(0, how_many_sessions_list[i_t]):
                                sessions_list.append(trials_list[i_t].session_data[j_s])
                        else:
                            # get sessions up to how_many, starting from the last session 
                            for j_s in range(trials_list[i_t].num_sessions - how_many_sessions_list[i_t],
                                             trials_list[i_t].num_sessions):
                                sessions_list.append(trials_list[i_t].session_data[j_s])
            else:
                raise ValueError('Trial', i_t, 'not found in self.trials. Missing trial is', trials_list[i_t])
        return sessions_list

    # returns number of sessions in the specified trials
    def get_num_sessions(self, trials_list):
        num_sessions_list = np.zeros(len(trials_list))
        for i_t in range(len(trials_list)):
            # Compare based on trial equality relation (date, time, hand, AND user_id equal)
            if trials_list[i_t] in self.trials:
                num_sessions_list[i_t] = trials_list[i_t].num_sessions
            else:
                raise ValueError('Trial', i_t, 'not found in self.trials. Missing trial is', trials_list[i_t])
        return num_sessions_list.astype(int)


# %% CLASS - TRIAL
# TODO: define data structure that defines what e.g. contact data means for a given trial type
# TODO: write function to check if any timestamp issues in data as we import it (e.g. a big jump in time)
# TODO: add method to auto-check contact arrays for quick pulses / errors during the import
# TODO: write function to remove a particular session and update the num_sessions
# TODO: define a better user prompt for new trials

class Trial(object):
    def __init__(self, file_path):

        known_trials_data = dict.fromkeys(
            ['tap', 'guitar_hero_tap_hold', 'passive_motion_using_phone', 'passive_motion_no_task'])
        known_trials_data['tap'] = ['skywalk', 'accelerometer', 'gyroscope', 'magnetometer', 'quaternion', 'contact',
                                    'user_prompt']
        known_trials_data['guitar_hero_tap_hold'] = ['skywalk', 'skywalk_power', 'accelerometer', 'gyroscope',
                                                     'magnetometer', 'quaternion', 'contact']
        known_trials_data['passive_motion_using_phone'] = ['skywalk', 'skywalk_power', 'accelerometer', 'gyroscope',
                                                           'magnetometer', 'quaternion']
        known_trials_data['passive_motion_no_task'] = ['skywalk', 'skywalk_power', 'accelerometer', 'gyroscope',
                                                       'magnetometer', 'quaternion']

        self.filepath = file_path

        with h5py.File(file_path, "r") as f:
            # Get list of all sessions (note: currently actual session IDs are arbitrary, so we relabel as 0, 1, 2...)
            sessions_list = list(f.keys())

            metadata = list(f['metadata'][()])
            # Remove metadata from sessions_list to ensure we don't iterate over it
            sessions_list.remove('metadata')
            sessions_list.remove('__DATA_TYPES__')

            # Verify trial type and its constituent data streams are known
            if metadata[0].decode('utf-8') not in known_trials_data:
                raise ValueError("Specified trial_type not a key in known_trials_data. Specified trial_type is:",
                                 metadata[0].decode('utf-8'), ". Known trials are:", list(known_trials_data.keys()),
                                 ". Either change trial_type or add new trial_type and data list to "
                                 "known_trials_data.")

            # Init trial metadata
            self.trial_type, self.user_id, self.firmware_version, self.hand, self.notes = \
                metadata[0].decode('utf-8'), metadata[1].decode('utf-8'), metadata[3].decode('utf-8'), \
                metadata[4].decode('utf-8'), metadata[5].decode('utf-8')
            self.date = datetime.strptime(metadata[2].decode('utf-8'), '%Y-%m-%dT%H-%M-%S').date()
            self.time = datetime.strptime(metadata[2].decode('utf-8'), '%Y-%m-%dT%H-%M-%S').time()

            # Init pandas structure for session data
            self.session_data = pd.DataFrame(columns=list(range(len(sessions_list))),
                                             index=known_trials_data[self.trial_type])
            self.num_sessions = len(sessions_list)

            # Save all session data to the Trial instance
            for i_s in range(len(sessions_list)):
                for data_stream in known_trials_data[self.trial_type]:
                    # Initialize column names for skywalk and IMU data
                    channel_names = None
                    if data_stream == 'skywalk':
                        channel_counter = list(
                            range(np.array(f[sessions_list[i_s]][(data_stream + '_data')][()]).shape[1]))
                        channel_names = ["CH" + str(x) for x in channel_counter]
                    elif data_stream == 'accelerometer' or data_stream == 'gyroscope' or \
                            data_stream == 'magnetometer':
                        channel_counter = ['x', 'y', 'z']
                        channel_names = [data_stream + '_' + x for x in channel_counter]
                    elif data_stream == 'quaternion':
                        channel_counter = ['a', 'b', 'c', 'd']
                        channel_names = [data_stream + '_' + x for x in channel_counter]
                    elif data_stream == 'user_prompt':
                        channel_names = ['swipe_direction', 'clicklocx', 'clicklocy', 'mode']

                    # Create the array without timestamps
                    self.session_data[i_s][data_stream] = pd.DataFrame(
                        np.array(f[sessions_list[i_s]][(data_stream + '_data')][()]), columns=channel_names)
                    # Add timestamps in the index
                    self.session_data[i_s][data_stream].index = np.array(
                        f[sessions_list[i_s]][(data_stream + '_timestamps')][()])

    def __repr__(self):
        return f'Trial <trial_type={self.trial_type}, user_id={self.user_id}, date={self.date}, ' \
               f'firmware_version={self.firmware_version}, hand={self.hand}, num_sessions={self.num_sessions}>'

    def __eq__(self, other):
        if type(other) is type(self):
            return (
                    self.date == other.date and self.time == other.time and self.user_id == other.user_id and
                    self.hand == other.hand)
        return False
