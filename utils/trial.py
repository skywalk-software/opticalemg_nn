from datetime import datetime as dt
import yaml
import h5py
import numpy as np
import pandas as pd

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
        
        self.srs = {}
        for k, v in self.data.items():
            if "_timestamps" in k and v is not None:
                self.srs[k.split("_")[0]] = 1000000 / (v[1:] - v[:-1]).mean()
