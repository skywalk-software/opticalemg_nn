import io
import math
import statistics
from typing import cast

import PIL
import matplotlib
import pandas as pd
import numpy as np
import torch
from matplotlib import pyplot as plt
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchsummary import summary
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sklearn import preprocessing
from torchvision.transforms import ToTensor

from functions_postprocessing import plot_predictions
from metrics import process_clicks

CLICK_REGION = 10
NONE_CLICK_REGION_WEIGHT = 2





def timeseries_from_sessions_list(
        sessions_list: [pd.Series],
        seq_length: int, fit_scaler=False, scaler_to_use=None, imu_data=None, shuffle=False):
    labels_array = np.empty((0,))
    data_array = np.empty((0, sessions_list[0]['skywalk'].shape[1]))
    for session in sessions_list:
        # Collapse labels onto skywalk timestamps
        labels_array = np.append(labels_array, np.array(session['contact'][0][session['skywalk'].index]), axis=0)
        partial_data_array = np.array(session['skywalk'])
        # add IMU data if applicable
        if imu_data is not None:
            for data_stream in imu_data:
                partial_data_array = np.append(partial_data_array, np.array(session[data_stream]), axis=1)
        data_array = np.append(data_array, np.array(session['skywalk']), axis=0)

    # Shifting indexing by seq_length-1 allows for prediction of the latest timestep's value
    # TODO shifting should occur on a per-array basis I believe -
    #  this shifting causes more issues at border between sessions
    data_array = data_array[:-(seq_length - 1)]
    labels_array = labels_array[(seq_length - 1):]

    # if fitting a new scale
    if fit_scaler:
        if scaler_to_use is not None:
            raise ValueError(
                "Cannot assign scaler and fit a new one! Either change fit_scaler to False or remove scaler_to_use.")
        scaler = preprocessing.StandardScaler().fit(data_array)
        data_array = scaler.transform(data_array)
        dataset = SkywalkDataset(data_array, labels_array, seq_length)
        return dataset, data_array, labels_array, scaler

    # If scaler was provided (e.g. this is test data)
    elif scaler_to_use is not None:
        data_array = scaler_to_use.transform(data_array)
        dataset = SkywalkDataset(data_array, labels_array, seq_length)
        return dataset, data_array, labels_array

    # Default, no scaler at all
    else:
        dataset = SkywalkDataset(data_array, labels_array, seq_length)
        return dataset, data_array, labels_array


