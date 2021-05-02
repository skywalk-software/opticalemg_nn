import logging

import h5py
import pytorch_lightning as pl
import torch.optim
from torch import nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import numpy as np

from skywalk_leap_dataset import SkywalkLeapDataset


# %%

def normalize(arr: np.array):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

def load_h5(filename: str):
    data_file = h5py.File(filename, 'r')

    leapmotion_timestamps, leapmotion_data, skywalk_timestamps, skywalk_data = \
        data_file["leapmotion_timestamps"], \
        data_file["leapmotion_data"], \
        data_file["skywalk_timestamps"], \
        data_file["skywalk_data"]

    leapmotion_timestamps, leapmotion_data, skywalk_timestamps, skywalk_data = np.array(
        leapmotion_timestamps), np.array(
        leapmotion_data), np.array(skywalk_timestamps), np.array(skywalk_data)

    leapmotion_data = np.array(leapmotion_data)
    leapmotion_data[:, 1] = np.abs(leapmotion_data[:, 1]) * 0
    leapmotion_data[:, 3] = normalize(leapmotion_data[:, 3])
    leapmotion_data[:, 4] = normalize(leapmotion_data[:, 4])

    train_val_div = len(leapmotion_timestamps) // 3 * 2
    train_dataset = SkywalkLeapDataset(
        1,
        leapmotion_timestamps[:train_val_div],
        leapmotion_data[:train_val_div],
        skywalk_timestamps,
        skywalk_data
    )

    val_dataset = SkywalkLeapDataset(
        1,
        leapmotion_timestamps[train_val_div:],
        leapmotion_data[train_val_div:],
        skywalk_timestamps,
        skywalk_data
    )
    return train_dataset, val_dataset


all_data_path = [
    "data/2021-04-08T22-56-18.h5",
    # "data/2021-04-26T23-20-38.h5",
    # "data/2021-04-26T23-22-49.h5",
    # "data/2021-04-26T23-24-36.h5"
]

all_data = list(zip(*[load_h5(path) for path in all_data_path]))

train_dataset, val_dataset = ConcatDataset(all_data[0]), ConcatDataset(all_data[1])

#%%
sw_idx, lm_idx = np.array(list(zip(*all_data[0][0].skywalk_idx_to_leapmotion_map)))

lm = normalize(np.sum(np.abs(np.diff(all_data[0][0].leapmotion_data[lm_idx], axis=0)), axis=1))
sw = normalize(np.sum(np.abs(np.diff(all_data[0][0].skywalk_data[sw_idx], axis=0)), axis=1))

#%%

import matplotlib.pyplot as plt

start = 1700
range_len = 1000
end = start + range_len

plt.plot(range(range_len), lm[start:end])
plt.show()

#%%

plt.plot(range(range_len), sw[start:end])
plt.show()

#%%
plt.plot(range(range_len), all_data[0][0].leapmotion_data[lm_idx][start:end])
plt.show()

# %%

class NnModel(pl.LightningModule):
    def __init__(self, learning_rate=0.001):
        super().__init__()
        self.learning_rate = learning_rate
        self.mlp_1 = nn.Sequential(
            nn.Linear(in_features=12, out_features=12),
            nn.ReLU()
        )
        self.mlp_2 = nn.Sequential(
            nn.Linear(in_features=12, out_features=12),
            nn.ReLU()
        )
        self.mlp_3 = nn.Sequential(
            nn.Linear(in_features=12, out_features=12),
            nn.ReLU()
        )
        self.mlp_4 = nn.Sequential(
            nn.Linear(in_features=12, out_features=8),
            nn.ReLU()
        )
        self.mlp_5 = nn.Sequential(
            nn.Linear(in_features=8, out_features=4)
        )
        self.loss = nn.L1Loss(reduction='none')

    def forward(self, x):
        x = x[:, -1, :]
        out = self.mlp_5(self.mlp_4(self.mlp_3(self.mlp_2(self.mlp_1(x)))))
        return out

    def loss_fn(self, out, target):
        return self.loss(out, target)

    def configure_optimizers(
            self,
    ):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.8)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = torch.mean(self.loss_fn(out, y[:, 1:]), dim=0)
        pitch_loss, yaw_loss, thumb_dist_loss, other_dist_loss = loss
        total_loss = torch.sum(loss, dim=0)
        self.log("train/loss", total_loss)
        self.log("train/pitch_loss", pitch_loss)
        self.log("train/yaw_loss", yaw_loss)
        self.log("train/thumb_dist_loss", thumb_dist_loss)
        self.log("train/other_dist_loss", other_dist_loss)
        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = torch.mean(self.loss_fn(out, y[:, 1:]), dim=0)
        pitch_loss, yaw_loss, thumb_dist_loss, other_dist_loss = loss
        total_loss = torch.sum(loss, dim=0)
        self.log("val/loss", total_loss)
        self.log("val/pitch_loss", pitch_loss)
        self.log("val/yaw_loss", yaw_loss)
        self.log("val/thumb_dist_loss", thumb_dist_loss)
        self.log("val/other_dist_loss", other_dist_loss)
        return total_loss


# %%

class SkywalkDataModel(pl.LightningDataModule):
    train_dataset: Dataset
    val_dataset: Dataset

    def __init__(self, batch_size=64):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)


# %%
logging.getLogger().setLevel(logging.INFO)

# checkpoint_callback = pl.callbacks.ModelCheckpoint(
#     monitor='valid_loss',
#     dirpath='./',
#     filename='models-{epoch:02d}-{valid_loss:.2f}',
#     save_top_k=3,
#     mode='min')

mod = NnModel(learning_rate=0.001)
# trainer = pl.Trainer(max_epochs=6, callbacks=[checkpoint_callback])
trainer = pl.Trainer(max_epochs=100, auto_lr_find=True)

#%%
# mod.load_from_checkpoint("nn.ckpt", learning_rate=0.001)
# trainer = pl.Trainer(resume_from_checkpoint="nn.ckpt", max_epochs=101)

trainer.fit(model=mod, datamodule=SkywalkDataModel())

# %%
trainer.save_checkpoint("nn2.ckpt")

#%%
mod = NnModel.load_from_checkpoint("nn.ckpt", learning_rate=0.001)

# %%
data_x, data_y = train_dataset[1234]
ans_y = mod(torch.Tensor([data_x]))
print(ans_y)
print(data_y)


# %%
SERIAL_PORT = "/dev/cu.usbmodem82125401"
SERIAL_BAUD_RATE = 115200

import serial

skywalk_serial = serial.Serial(port=SERIAL_PORT, baudrate=SERIAL_BAUD_RATE)

while True:
    line = skywalk_serial.readline()
    decoded_input = [float(item) for item in line.decode('utf-8').split(",")[:-1]]
    if len(decoded_input) != 12:
        continue
    input_tensor = torch.tensor([[decoded_input]])
    output_tensor = mod(input_tensor)
    print(output_tensor)

