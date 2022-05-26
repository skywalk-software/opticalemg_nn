import math
import statistics
from typing import cast

import pandas as pd
import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchsummary import summary
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sklearn import preprocessing

from metrics import process_clicks


class SkywalkDataset(Dataset):

    def __init__(self, data_array: np.ndarray, labels_array: np.ndarray, seq_length: int):
        self.data_array = data_array.astype(np.float32)
        self.labels_array = labels_array.astype(np.long)
        assert data_array.shape[0] == labels_array.shape[0]
        self.seq_length = seq_length
        self.data_length = len(data_array) - seq_length

    def __len__(self):
        return self.data_length

    def __getitem__(self, item):
        return self.data_array[item: item + self.seq_length], self.labels_array[item]


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

    # if fitting a new scaler
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


class SkywalkCnnV1(pl.LightningModule):
    def __init__(self, kernel_size: int, in_channels: int, seq_length: int, test_dataset_names: [str]):
        super(SkywalkCnnV1, self).__init__()
        self.test_dataset_names = test_dataset_names
        self.kernel_size = kernel_size
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, 48, kernel_size,),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(48),
            nn.Flatten(),
            nn.Linear(144, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 2),
        )
        for module in self.layers.modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d):
                nn.init.xavier_uniform_(module.weight.data)
                # module.weight.data.fill_(1)
                module.bias.data.fill_(0)
        self.loss = nn.CrossEntropyLoss()
        self.hparams.lr = 0.001

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.transpose(x, 1, 2)
        x = self.layers(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = ReduceLROnPlateau(optimizer, 'min')
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train/loss",
                "frequency": 1
            },
        }

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, dataset_idx):
        x, y = batch
        y_hat = self(x)
        # y_same = torch.vstack([1 - y, y]).T
        return y.cpu(), y_hat.detach().cpu()

    def validation_epoch_end(self, outputs):
        tensorboard = cast(TensorBoardLogger, self.logger).experiment

        val_prefix = "val/"
        val_loss_total = 0
        val_acc_total = 0
        val_samples = 0

        val_total_true_clicks = 0
        val_total_false_clicks = 0
        val_total_missed_clicks = 0
        val_total_detected_clicks = 0
        val_on_set_offsets = []
        val_off_set_offsets = []
        val_drops = []

        for dataset_idx, dataset_outputs in enumerate(outputs):
            dataset_prefix = f"val-all/{self.test_dataset_names[dataset_idx]}/"
            y_list = []
            y_hat_list = []
            for batch_idx, (y, y_hat) in enumerate(dataset_outputs):
                y_list += [y]
                y_hat_list += [y_hat]
            total_y = torch.cat(y_list)
            total_y_hat = torch.cat(y_hat_list)

            assert len(total_y) == len(total_y_hat)

            samples = len(total_y)

            loss = self.loss(total_y_hat, total_y)
            estimated_y = (total_y_hat[:, 0] < total_y_hat[:, 1]).long()
            accuracy = torch.sum((estimated_y == total_y)) / samples

            result = process_clicks(total_y, estimated_y)
            total_true_clicks, total_false_clicks, total_missed_clicks, total_detected_clicks, \
                on_set_offsets, off_set_offsets, drops = result
            assert len(on_set_offsets) == len(off_set_offsets) == total_detected_clicks
            self.log(f"{dataset_prefix}loss", loss)
            self.log(f"{dataset_prefix}accuracy", accuracy)
            self.log(f"{dataset_prefix}FP-P", math.nan if total_true_clicks == 0 else total_false_clicks / total_true_clicks)
            self.log(f"{dataset_prefix}TP-P", math.nan if total_true_clicks == 0 else total_detected_clicks / total_true_clicks)
            self.log(f"{dataset_prefix}drops-P", math.nan if total_true_clicks == 0 else len(drops) / total_true_clicks)
            self.log(f"{dataset_prefix}std-onset", math.nan if total_detected_clicks < 2 else statistics.stdev(on_set_offsets))
            self.log(f"{dataset_prefix}std-offset", math.nan if total_detected_clicks < 2 else statistics.stdev(off_set_offsets))

            if total_detected_clicks > 1:
                tensorboard.add_histogram(f"{dataset_prefix}onset", np.array(on_set_offsets), self.current_epoch)
                tensorboard.add_histogram(f"{dataset_prefix}offset", np.array(off_set_offsets), self.current_epoch)

            if len(drops) > 1:
                tensorboard.add_histogram(f"{dataset_prefix}drops", np.array(drops), self.current_epoch)

            val_samples += samples
            val_loss_total += loss * samples
            val_acc_total += accuracy * samples

            val_total_true_clicks += total_true_clicks
            val_total_false_clicks += total_false_clicks
            val_total_missed_clicks += total_missed_clicks
            val_total_detected_clicks += total_detected_clicks
            val_on_set_offsets += on_set_offsets
            val_off_set_offsets += off_set_offsets
            val_drops += drops

        val_loss = val_loss_total / val_samples
        val_acc = val_acc_total / val_samples

        self.log(f"{val_prefix}loss", val_loss)
        self.log(f"{val_prefix}accuracy", val_acc)
        self.log(f"{val_prefix}FP-P", math.nan if val_total_true_clicks == 0 else val_total_false_clicks / val_total_true_clicks)
        self.log(f"{val_prefix}TP-P", math.nan if val_total_true_clicks == 0 else val_total_detected_clicks / val_total_true_clicks)
        self.log(f"{val_prefix}drops-P", math.nan if val_total_true_clicks == 0 else len(val_drops) / val_total_true_clicks)
        self.log(f"{val_prefix}std-onset", math.nan if val_total_detected_clicks < 2 else statistics.stdev(val_on_set_offsets))
        self.log(f"{val_prefix}std-offset", math.nan if val_total_detected_clicks < 2 else statistics.stdev(val_off_set_offsets))

        if val_total_detected_clicks > 1:
            tensorboard.add_histogram(f"{val_prefix}onset", np.array(val_on_set_offsets), self.current_epoch)
            tensorboard.add_histogram(f"{val_prefix}offset", np.array(val_off_set_offsets), self.current_epoch)
        if len(val_drops) > 1:
            tensorboard.add_histogram(f"{val_prefix}drops", np.array(val_drops), self.current_epoch)
        return val_loss


def get_predictions(model, test_dataset_for_pred):
    output_list = []
    for x, y in test_dataset_for_pred:
        output = model(torch.Tensor(x)).detach().numpy()
        output_list += [output]
    output_array = np.concatenate(output_list, axis=0)
    return np.argmax(output_array, axis=1)
