from tqdm import tqdm
from boltons.fileutils import iter_find_files
import pandas as pd
import hydra
from omegaconf import OmegaConf
import logging
import yaml
import os
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torch.utils.data import random_split
from sklearn import preprocessing
from functools import partial
from datetime import datetime as dt
import yaml
import h5py

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
        self.data = {}
        self.nir_srs = {}
        with h5py.File(self.filepath, "r") as f:
            keys = list(f.keys())
            keys.remove('metadata')
            keys.remove('__DATA_TYPES__')
            self.sessions_list = keys
            for sess in self.sessions_list:
                session_data = f[sess]
                self.data[sess] = dict((k, np.array(session_data[k])) for k in session_data.keys())
                for k, v in self.data[sess].items():
                    if 0 in v.shape:
                        self.data[sess][k] = None
        
            ## pull some numbers on inconsistencies in sampling rate
            for sess, data in self.data.items():
                v = data['skywalk_timestamps']
                self.nir_srs[sess] = (v[1:] - v[:-1]).mean() / 1000

    @staticmethod
    def trial_stats(trials, stream, seq_len, stride=1):
        data_arr = []
        stream = stream + '_data'
        for t in tqdm(trials):
            for sess, data in t.data.items():
                if data is None:
                    continue
                for j in range(0, data[stream].shape[0] - seq_len, stride):
                    data_arr.append(data[stream][j:j+seq_len].tolist())
        data_arr = np.concatenate(data_arr)
        print(data_arr.shape)
        return data_arr.mean(0), data_arr.std(0)

class SessionDataset(Dataset):
    def __init__(
        self, name, data, metadata, seq_len, stride, data_stream, contact_channel
    ):
        self.name = name
        self.stride = stride
        self.seq_len = seq_len
        self.data_stream = data_stream
        self.data = data
        self.nir_d = self.data[f"{self.data_stream}_data"]
        self.nir_ts = self.data[f"{self.data_stream}_timestamps"]
        self.contact_d = self.data["contact_data"]
        self.contact_ts = self.data["contact_timestamps"]
        self.contact_channel = contact_channel
        self.data_idxs = list(range(0, self.nir_d.shape[0] - self.seq_len, self.stride))
        self.metadata = metadata

    def __len__(self):
        return len(self.data_idxs)
    
    def __getitem__(self, idx):
        start = self.data_idxs[idx]
        end = start + self.seq_len
        data = torch.from_numpy(self.nir_d[start:end, :])
        return data.permute(1, 0)

class SkywalkDataset(Dataset):
    def __init__(self, trials, seq_length, stride, data_stream, contact_channel, norm=None):
        self.data_stream = data_stream
        self.seq_length = seq_length
        self.stride = stride
        self.contact_channel = contact_channel
        self.trials = trials
        self.norm = norm
        datasets = []
        for trial in self.trials:
            for sess, data in trial.data.items():
                if data is None:
                    continue
                datasets.append(
                    SessionDataset(
                        sess, data, trial.metadata, 
                        self.seq_length, self.stride, self.data_stream, self.contact_channel
                    )
                )
        self.dataset = ConcatDataset(datasets)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.norm:
            return self.norm(data=self.dataset[idx])
        else:
            return self.dataset[idx]

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
    if "exclude_files" in config and config["exclude_files"] is not None:
        condition &= ~df["filename"].isin(config["exclude_files"])
    if condition is None:
        logger.debug('No metadata filters appplied')
        return df
    return df[condition]

def select_data_files(config, name):
    df = build_data_frame(config['dirpaths'])
    if df is None:
        logger.debug('No data files found for data config {}'.format(name))
        return None
    df = filter_df(config, df)
    files = df['filename'].tolist()
    dirs = df['directory'].tolist()
    if len(files) == 0:
        logger.debug('No data files found for data config {}'.format(name))
        return None
    files = [os.path.join(d, f + ".h5") for d, f in zip(dirs, files)]
    logger.info('Found {} data files for data config {}'.format(len(files), name))
    return files

def get_files_from_all(cfgs):
    files_dict = {}
    logger.info('Collecting files from data configs: {}'.format(",".join(list(cfgs.keys()))))
    for name, cfg in cfgs.items():
        cfg_files = select_data_files(cfg, name)
        if cfg_files is None:
            cfg_files = []
        files_dict[name] = cfg_files
    logger.info('Found {} files TOTAL'.format(sum([len(v) for v in files_dict.values()])))
    for k, v in files_dict.items():
        files_dict[k] = list(set(v))
    allfiles = []
    for k, v in files_dict.items():
        if v is None:
            continue
        allfiles += v
    allfiles = list(set(allfiles))
    return files_dict, allfiles

def standardize(data, mu, sigma):
    return (data - mu[:, None]) / sigma[:, None]

def get_datasets(data_cfg, dataset_cfg):
    _, allfiles = get_files_from_all(data_cfg)
    trials = [Trial(f) for f in allfiles]

    if dataset_cfg.normalize:
        mu, sigma = Trial.trial_stats(
            trials, dataset_cfg.stream, dataset_cfg.seq_length, dataset_cfg.stride)
        norm = partial(standardize, mu=mu, sigma=sigma)
    else:
        norm = None

    dataset = SkywalkDataset(
        trials, 
        dataset_cfg.seq_length, 
        dataset_cfg.stride, 
        dataset_cfg.stream, 
        dataset_cfg.contact_channel,
        norm=norm
    )
    train, val, test = random_split(
        dataset, [dataset_cfg.train_percent, dataset_cfg.val_percent, dataset_cfg.test_percent])
    
    return train, val, test

def get_dataloaders(data_cfg, dataset_cfg, dataloader_cfg):
    train, val, test = get_datasets(data_cfg, dataset_cfg)
    train_loader = DataLoader(
        train, batch_size=dataloader_cfg.batch_size, shuffle=True, pin_memory=dataloader_cfg.pin_memory,)
    val_loader = DataLoader(
        val, batch_size=dataloader_cfg.batch_size, shuffle=False, pin_memory=dataloader_cfg.pin_memory,)
    test_loader = DataLoader(
        test, batch_size=dataloader_cfg.batch_size, shuffle=False, pin_memory=dataloader_cfg.pin_memory,)
    return train_loader, val_loader, test_loader


@hydra.main(version_base="1.2.0", config_path="../config", config_name="conf")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))







if __name__ == '__main__':
    main()

