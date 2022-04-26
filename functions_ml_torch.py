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
        return self.data_array[item: item + self.seq_length], self.labels_array[item + self.seq_length]


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
            nn.Softmax()
        )
        self.loss = nn.CrossEntropyLoss()
        self.hparams.lr = 1e-03


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
        accuracy = torch.sum(((y_hat[:, 0] < y_hat[:, 1]).int() == y)) / len(x)
        loss = self.loss(y_hat, y)
        self.log(f"val/{self.test_dataset_names[dataset_idx]}/loss", loss)
        self.log(f"val/{self.test_dataset_names[dataset_idx]}/accuracy", accuracy)
        return loss


def get_predictions(model, test_dataset_for_pred):
    output_list = []
    for x, y in test_dataset_for_pred:
        output = model(torch.Tensor(x)).detach().numpy()
        output_list += [output]
    output_array = np.concatenate(output_list, axis=0)
    return np.argmax(output_array, axis=1)
