import logging

import h5py
import pytorch_lightning as pl
import torch.optim
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np

from skywalk_leap_dataset import SkywalkLeapDataset

# %%

data_file = h5py.File('./data/2021-04-08T22-56-18.h5', 'r')

leapmotion_timestamps, leapmotion_data, skywalk_timestamps, skywalk_data = \
    data_file["leapmotion_timestamps"], \
    data_file["leapmotion_data"], \
    data_file["skywalk_timestamps"], \
    data_file["skywalk_data"]


# %% normalize distance: I haven't figure our why, but the loss won't converge when I normalize the distance

def normalize(arr: np.array):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


leapmotion_data = np.array(leapmotion_data)
# leapmotion_data[:, 3] = normalize(leapmotion_data[:, 3])
# leapmotion_data[:, 4] = normalize(leapmotion_data[:, 4])

# %%
train_val_div = 20000
train_dataset = SkywalkLeapDataset(
    1,
    leapmotion_timestamps[:train_val_div],
    leapmotion_data[:train_val_div],
    skywalk_timestamps[:train_val_div],
    skywalk_data[:train_val_div]
)

val_dataset = SkywalkLeapDataset(
    1,
    leapmotion_timestamps[train_val_div:],
    leapmotion_data[train_val_div:],
    skywalk_timestamps[train_val_div:],
    skywalk_data[train_val_div:]
)


# %%

class NnModel(pl.LightningModule):
    def __init__(self, learning_rate):
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
            nn.Linear(in_features=8, out_features=4),
            nn.ReLU()
        )
        self.loss = nn.L1Loss(reduction='none')

    def forward(self, x):
        x = x[:, -1, :]
        out = self.mlp_5(self.mlp_4(self.mlp_3(self.mlp_2(self.mlp_1(x)))))
        return out

    def loss_fn(self, out, target):
        return self.loss(out, target[:, 1:])

    def configure_optimizers(
            self,
    ):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.8)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = torch.mean(self.loss_fn(out, y), dim=0)
        pitch_loss, yaw_loss, thumb_dist_loss, other_dist_loss = loss
        total_loss = torch.mean(loss, dim=0)
        self.log("train/loss", total_loss)
        self.log("train/pitch_loss", pitch_loss)
        self.log("train/yaw_loss", yaw_loss)
        self.log("train/thumb_dist_loss", thumb_dist_loss)
        self.log("train/other_dist_loss", other_dist_loss)
        return thumb_dist_loss + other_dist_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = torch.mean(self.loss_fn(out, y), dim=0)
        pitch_loss, yaw_loss, thumb_dist_loss, other_dist_loss = loss
        total_loss = torch.mean(loss, dim=0)
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

trainer.fit(model=mod, datamodule=SkywalkDataModel())
