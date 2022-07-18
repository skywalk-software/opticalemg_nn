from boltons.fileutils import iter_find_files
import pandas as pd
import hydra
from omegaconf import OmegaConf
import logging
import yaml
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from datetime import datetime as dt
import yaml
import h5py

## ignore guitar-her-v1 data


logger = logging.getLogger(__name__)

CLICK_REGION = 10
NONE_CLICK_REGION_WEIGHT = 2

metadata_keymap = {
    'user': 'users',
    'datacollector_version': 'datacollector_versions',
    'firmware_version': 'firmware_versions',
    'date': 'dates',
    'hand': 'hands',
    'trial_type': 'trial_types',
}

def build_data_frame(dirpaths):
    all_yamls = []
    for d in dirpaths:
        found = iter_find_files(directory=d, patterns='*.yaml', include_dirs=True)
        found = [(os.path.basename(f), os.path.dirname(f)) for f in found]
        all_yamls += found
    if len(all_yamls) == 0:
        return None
    df = []
    for f, d in all_yamls:
        filepath = os.path.join(d, f)
        with open(filepath, 'r') as f:
            yaml_data = yaml.safe_load(f)
            yaml_data["directory"] = d
            df.append(yaml_data)
    df = pd.DataFrame(df)
    return df

def filter_df(config, df):
    condition = None
    for k, v in metadata_keymap.items():
        if config[v] is not None:
            if condition is None:
                condition = df[k].isin(config[v])
            else:
                condition &= df[k].isin(config[v])
    if "exclude_files" in config:
        condition &= ~df["filename"].isin(config["exclude_files"])
    if condition is None:
        logger.debug('No metadata filters appplied')
        return df
    return df[condition]

def select_data_files(config):
    df = build_data_frame(config['dirpaths'])
    if df is None:
        logger.debug('No data files found')
        return None
    df = filter_df(config, df)
    files = df['filename'].tolist()
    dirs = df['directory'].tolist()
    if len(files) == 0:
        logger.debug('No data files found')
        return None
    files = [os.path.join(d, f + ".h5") for d, f in zip(dirs, files)]
    logger.debug('Found {} data files'.format(len(files)))
    return files

def get_files_from_all(cfgs):
    files = {}
    logger.info('Collecting files from all dataset configs')
    for name, cfg in cfgs.items():
        cfg_files = select_data_files(cfg)
        if cfg_files is None:
            cfg_file = []
        files[name] = cfg_files
    logger.info('Found {} files'.format(len(files)))
    return list(set(files))


## updated to parse out multiple sessions if there are multiple sessions
class Trial(object):
    def __init__(self, file_path):
        self.filepath = file_path
        self.load_metadata()
        self.process_h5()

    def load_metadata(self):
        self.metadata = yaml.safe_load(open(self.filepath.replace('h5', 'yaml'), 'r'))
        self.metadata["date"] = dt.strptime(self.metadata['date'], '%Y-%m-%d').date()
        self.metadata["time"] = dt.strptime(self.metadata['time'], '%H:%M:%S').time()
        self.datacollector_version, self.user, self.firmware_version, self.hand, self.notes, self.date, self.time = \
            self.metadata['datacollector_version'], self.metadata['user'], self.metadata['firmware_version'], \
            self.metadata['hand'], self.metadata['notes'], self.metadata["date"], self.metadata["time"]

    def process_h5(self):
        with h5py.File(self.filepath, "r") as f:
            keys = list(f.keys())
            keys.remove('metadata')
            keys.remove('__DATA_TYPES__')
            self.data = dict((k, np.array(f[k])) for k in keys)
        for k, v in self.data.items():
            if 0 in v.shape:
                self.data[k] = None
        
        ## pull some numbers on inconsistencies in sampling rate
        self.srs = {}
        for k, v in self.data.items():
            if "_timestamps" in k and v is not None:
                self.srs[k.split("_")[0]] = 1000000 / (v[1:] - v[:-1]).mean()

    def get_data_idxs(self, key, seq_length, stride):
        if self.data[key] is None:
            return None
        return np.arange(0, self.data[key].shape[0] - seq_length, stride)

    def get_data(self, idx, key, seq_length):
        if self.data[key] is None:
            return None
        return self.data[key][idx:idx + seq_length]


## contact data is touch pad in click glove
## contact 3 channels --> index, middle, ring finger click. boolean.

## specific dataset for each session
## torch.ConcatDataset 
class SkywalkDataset(Dataset):
    def __init__(self, trials, seq_length, stride=1):
        self.data_key = "skywalk"
        self.seq_length = seq_length
        self.stride = stride

        self.trials = trials
        self.idx_to_trial = []
        self.trial_base = []
        self.data_idxs = {}
        for i,t in enumerate(self.trials):
            self.trial_base_idx = len(self.idx_to_trial)
            data_idxs = t.get_data_idxs(self.data_key, self.seq_length, self.stride)
            self.idx_to_trial += [i for _ in range(len(data_idxs))]
            self.data_idxs[i] = data_idxs

    def __len__(self):
        return len(self.trial_idxs)

    def __getitem__(self, idx):
        trial_idx = self.trial_idxs[idx]
        trial = self.trials[trial_idx]
        data_idx = self.data_idxs[trial_idx][idx % len(self.data_idxs[trial_idx])]
        return trial.get_data(idx, self.data_key, self.seq_length)


    def __init__(self, data_array: np.ndarray, labels_array: np.ndarray, seq_length: int):
        self.data_array = torch.FloatTensor(data_array)
        self.labels_array = torch.LongTensor(labels_array)
        assert data_array.shape[0] == labels_array.shape[0]
        self.seq_length = seq_length
        self.data_length = len(data_array) - seq_length
        self.weights_array = torch.ones(self.labels_array.shape, dtype=torch.float)
        for i in range(self.weights_array.shape[0]):
            start = max(i - CLICK_REGION, 0)
            end = min(i + CLICK_REGION, len(self.labels_array) - 1)
            if not torch.any(self.labels_array[start: end]):
                self.weights_array[i] = NONE_CLICK_REGION_WEIGHT

    def __len__(self):
        return self.data_length

    def __getitem__(self, item):
        return self.data_array[item: item + self.seq_length], self.labels_array[item], self.weights_array[item]


@hydra.main(version_base="1.2.0", config_path="../config", config_name="conf")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    datafiles = get_files_from_all(cfg.datasets)
    trials = [Trial(f) for f in datafiles]

if __name__ == '__main__':
    main()

