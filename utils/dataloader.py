from tqdm import tqdm
from boltons.fileutils import iter_find_files
import pandas as pd
import logging
import yaml
import os
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torch.utils.data import random_split as torch_random_split
from functools import partial
from datetime import datetime as dt
import yaml
import h5py

logger = logging.getLogger(__name__)

CLICK_REGION_DIST = 10
NONE_CLICK_REGION_WEIGHT = torch.LongTensor([2])
CLICK_REGION_WEIGHT = torch.LongTensor([1])

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

    ## This can + should be optimized or done offline
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
        self.contact_ts = self.data["contact_timestamps"]
        self.contact_idxs = np.searchsorted(self.contact_ts, self.nir_ts)
        self.contact_channel = contact_channel
        self.contact_d = self.data["contact_data"][self.contact_idxs][(seq_len-1):, contact_channel]
        self.contact_ts = self.contact_ts[self.contact_idxs][(seq_len-1):]
        self.data_idxs = list(range(0, self.nir_d.shape[0] - self.seq_len + 1, self.stride))
        self.metadata = metadata

    def __len__(self):
        return len(self.data_idxs)

    def get_weight(self, idx):
        start = max(idx - CLICK_REGION_DIST, 0)
        end = min(idx + CLICK_REGION_DIST, self.nir_ts.shape[0])
        if (self.contact_d[start:end] == 0).all():
            return NONE_CLICK_REGION_WEIGHT
        else:
            return CLICK_REGION_WEIGHT

    def __getitem__(self, idx):
        start = self.data_idxs[idx]
        end = start + self.seq_len
        data = torch.from_numpy(self.nir_d[start:end, :])
        data = data.permute(1, 0)
        data_ts = torch.from_numpy(self.nir_ts[start:end]).float()
        label = torch.LongTensor([self.contact_d[start]])
        label_ts = torch.Tensor([self.contact_ts[start]])
        weight = self.get_weight(start)
        meta = self.metadata["trial_type"]
        assert (data_ts[-1] == label_ts[0]).item()
        return data, label, weight, data_ts, label_ts, meta

class SkywalkDataset(Dataset):
    def __init__(self, trials, seq_length, stride, data_stream, contact_channel, stand=None):
        self.data_stream = data_stream
        self.seq_length = seq_length
        self.stride = stride
        self.contact_channel = contact_channel
        self.trials = trials
        self.stand = stand
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
        if self.stand:
            return self.stand(data=self.dataset[idx])
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
    if "exclude_files" in config and config["exclude_files"] is not None\
        and len(config["exclude_files"]) > 0:
        condition &= ~df["filename"].isin(config["exclude_files"])
    if condition is None:
        logger.debug('No metadata filters appplied')
        return df
    return df[condition]

def select_data_files(config, name):
    df = build_data_frame(config['dirpaths'])
    if df is None:
        logger.debug('No H5 files found for data config: {}'.format(name))
        return None
    df = filter_df(config, df)
    files = df['filename'].tolist()
    dirs = df['directory'].tolist()
    if len(files) == 0:
        logger.debug('No H5 files found for data config: {}'.format(name))
        return None
    files = [os.path.join(d, f + ".h5") for d, f in zip(dirs, files)]
    logger.info('Found {} H5 files for data config {}'.format(len(files), name))
    return files

def get_files_from_all(cfgs):
    files_dict = {}
    logger.info('Collecting files from data configs: {}'.format(list(cfgs.keys())))
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

def random_split(dataset, split_fractions):
    split = {}
    for k, v in split_fractions.items():
        dataset, split[k] = torch_random_split(dataset, [len(dataset) - v, v])
    return split

def get_datasets(data_cfg, dataset_cfg):
    files_by_cfg, _ = get_files_from_all(data_cfg)
    trials = {
        name: [Trial(f) for f in files_by_cfg[name]] for name in files_by_cfg.keys()
    }

    if dataset_cfg.standardize:
        logger.info("Computing statisitics for standardization... this may take a while")
        all_trials = []
        for name, trials in trials.items():
            all_trials += trials
        logger.info('Standardizing data')
        mu, sigma = Trial.trial_stats(
            all_trials, dataset_cfg.stream, dataset_cfg.seq_length, dataset_cfg.stride)
        stand = partial(standardize, mu=mu, sigma=sigma)
    else:
        logger.info("Skipping standardization")
        stand = None

    all_train = []
    all_val = []
    all_test = []

    for name, trials in trials.items():
        logger.info('Building dataset for data config: {}'.format(name))
        dataset = SkywalkDataset(
            trials, 
            dataset_cfg.seq_length, 
            dataset_cfg.stride, 
            dataset_cfg.stream, 
            dataset_cfg.contact_channel,
            stand=stand
        )
        this_cfg = data_cfg[name]
        logger.info("Train, val, test splits for {} are {}, {}, {}".format(
            name, this_cfg.train_percent, this_cfg.val_percent, this_cfg.test_percent))

        train_count, val_count, test_count = list(map(lambda x : round(x * len(dataset)),\
                [this_cfg.train_percent, this_cfg.val_percent, this_cfg.test_percent]))

        splits = random_split(
            dataset, {"train": train_count, "val": val_count, "test": test_count})

        train, val, test = splits["train"], splits["val"], splits["test"]

        logger.info('Size of {} train, val, test sets: {}, {}, {}'.format(
            name, len(train), len(val), len(test)))

        all_train.append(train)
        all_val.append(val)
        all_test.append(test)
    
    train = ConcatDataset(all_train)
    val = ConcatDataset(all_val)
    test = ConcatDataset(all_test)

    logger.info('TOTAL Size of train, val, test sets: {} {} {}'.format(len(train), len(val), len(test)))

    return train, val, test


def get_dataloaders(data_cfg, dataset_cfg, dataloader_cfg):
    train, val, test = get_datasets(data_cfg, dataset_cfg)
    logger.info('Building dataloaders')
    logger.info("Batch size is: {}".format(dataloader_cfg.batch_size))
    logger.info("Num workers is: {}".format(dataloader_cfg.num_workers))
    logger.info("Pin memory is: {}".format(dataloader_cfg.pin_memory))
    train_loader = DataLoader(
        train, batch_size=dataloader_cfg.batch_size, shuffle=True, 
        pin_memory=dataloader_cfg.pin_memory, num_workers=dataloader_cfg.num_workers)
    val_loader = DataLoader(
        val, batch_size=dataloader_cfg.batch_size, shuffle=False, 
        pin_memory=dataloader_cfg.pin_memory, num_workers=dataloader_cfg.num_workers)
    test_loader = DataLoader(
        test, batch_size=dataloader_cfg.batch_size, shuffle=False, 
        pin_memory=dataloader_cfg.pin_memory, num_workers=dataloader_cfg.num_workers)
    return train_loader, val_loader, test_loader
