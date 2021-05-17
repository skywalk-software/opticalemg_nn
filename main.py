import logging
import math

import h5py
import pytorch_lightning as pl
import torch.optim
from torch import nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import numpy as np

from skywalk_leap_dataset import SkywalkLeapDataset


# %%

def normalize(arr: np.array, min_val=None, max_val=None):
    if torch.is_tensor(arr):
        if min_val is None:
            min_val = torch.quantile(arr, .01)
        if max_val is None:
            max_val = torch.quantile(arr, .99)
    else:
        if min_val is None:
            min_val = np.quantile(arr, .01)
        if max_val is None:
            max_val = np.quantile(arr, .99)
    return (arr - min_val) / (max_val - min_val)

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

    skywalk_data = np.array(skywalk_data)
    for i in range(skywalk_data.shape[1]):
        skywalk_data[:, i] = normalize(skywalk_data[:, i])

    leapmotion_data = np.array(leapmotion_data)
    # leapmotion_data[:, 1] = (leapmotion_data[:, 1] > - math.pi / 4) * leapmotion_data[:, 1] - (leapmotion_data[:, 1] <= - math.pi / 4) * leapmotion_data[:, 1]
    leapmotion_data[:, 3] = 0 * normalize(leapmotion_data[:, 3])
    leapmotion_data[:, 4] = 0 * normalize(leapmotion_data[:, 4])

    train_val_div = len(skywalk_timestamps) // 4 * 3
    train_dataset = SkywalkLeapDataset(
        1,
        leapmotion_timestamps,
        leapmotion_data,
        skywalk_timestamps[1500:train_val_div],
        skywalk_data[1500:train_val_div]
    )

    val_dataset = SkywalkLeapDataset(
        1,
        leapmotion_timestamps,
        leapmotion_data,
        skywalk_timestamps[train_val_div:],
        skywalk_data[train_val_div:]
    )
    return train_dataset, val_dataset


all_data_path = [
    # "data/2021-04-08T22-56-18.h5",
    # "data/2021-04-26T23-20-38.h5",
    # "data/2021-04-26T23-22-49.h5",
    # "data/2021-04-26T23-24-36.h5"
    # "data2/2021-05-02T17-20-26.h5",
    # "data2/2021-05-02T17-42-48.h5"
    # "data3/2021-05-11T02-17-57.h5"
    # "data3/2021-05-11T03-14-49.h5"
    # "data3/2021-05-11T03-26-26.h5"
    # "data3/2021-05-11T03-37-12.h5"
    # "data3/2021-05-11T21-37-38.h5"
    "data4/2021-05-16T15-56-44.h5"
]
all_data = list(zip(*[load_h5(path) for path in all_data_path]))

train_dataset, val_dataset = ConcatDataset(all_data[0]), ConcatDataset(all_data[1])

#%%
sw_idx, lm_idx = np.array(list(zip(*all_data[0][0].skywalk_idx_to_leapmotion_map)))

lm = normalize(np.sum(np.abs(np.diff(all_data[0][0].leapmotion_data[lm_idx], axis=0)), axis=1))
sw = normalize(np.sum(np.abs(np.diff(all_data[0][0].skywalk_data[sw_idx], axis=0)), axis=1))

#%%

import matplotlib.pyplot as plt

start = 1300
range_len = 1000
end = start + range_len

plt.plot(range(range_len), lm[start:end])
plt.show()

#%%

plt.plot(range(range_len), sw[start:end])
plt.show()

#%%
lm_labels = ["event.valid", "event.pitch", "event.yaw", "event.thumbDist", "event.otherDist"]
for i in range(5):
    plt.plot(range(range_len), all_data[0][0].leapmotion_data[lm_idx][start:end][:, i], label=lm_labels[i])
plt.legend()
plt.show()

#%%
for i in range(12):
    plt.plot(range(range_len), all_data[0][0].skywalk_data[sw_idx][start:end][:, i], label=f"channel {i}")
plt.legend()
plt.show()

#%%
start = 0
range_len = 2000
end = start + range_len

input_group = []
output_group = []
for idx in range(start, end):
    input_group += [train_dataset[idx][0][0]]
    output_group += [train_dataset[idx][1][0]]
input_np = np.array(input_group)
output_np = np.array(output_group)

lm_labels = ["event.valid", "event.pitch", "event.yaw", "event.thumbDist", "event.otherDist"]
for i in range(5):
    plt.plot(range(range_len), output_np[:, i], label=lm_labels[i])
plt.legend()
plt.show()

for i in range(12):
    plt.plot(range(range_len), input_np[:, i], label=f"channel {i}")
plt.ylim((-0.5, 1.5))
plt.legend()
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
        out = out.unsqueeze(1).repeat(1, y.shape[1], 1)
        loss = torch.mean(self.loss_fn(out, y[:, :, 1:]), dim=0)
        pitch_loss, yaw_loss, thumb_dist_loss, other_dist_loss = torch.min(loss, dim=0)[0]
        total_loss = pitch_loss + yaw_loss + thumb_dist_loss + other_dist_loss
        self.log("train/loss", total_loss)
        self.log("train/pitch_loss", pitch_loss)
        self.log("train/yaw_loss", yaw_loss)
        self.log("train/thumb_dist_loss", thumb_dist_loss)
        self.log("train/other_dist_loss", other_dist_loss)
        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        out = out.unsqueeze(1).repeat(1, y.shape[1], 1)
        loss = torch.mean(self.loss_fn(out, y[:, :, 1:]), dim=0)
        pitch_loss, yaw_loss, thumb_dist_loss, other_dist_loss = torch.min(loss, dim=0)[0]
        total_loss = pitch_loss + yaw_loss + thumb_dist_loss + other_dist_loss
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
trainer.save_checkpoint("nn4.ckpt")

#%%
mod = NnModel.load_from_checkpoint("nn.ckpt", learning_rate=0.001)

# %%
np.set_printoptions(precision=4, suppress=True)

for idx in [100, 200, 300, 400, 500]:
    data_x, data_y = train_dataset[idx]
    ans_y = mod(torch.Tensor([data_x]))
    print("out", np.array(ans_y.detach().numpy()[0]))
    print("ans", data_y[0][1:])


# %%
SERIAL_PORT = "/dev/cu.usbmodem82125401"
SERIAL_BAUD_RATE = 115200

import serial

skywalk_serial = serial.Serial(port=SERIAL_PORT, baudrate=SERIAL_BAUD_RATE)

calibrated = False
skywalk_min = None
skywalk_max = None

aggregated_input = []
while True:
    line = skywalk_serial.readline()
    decoded_input = [float(item) for item in line.decode('utf-8').split(",")[:-1]]
    if len(decoded_input) != 12:
        print(decoded_input)
        continue
    aggregated_input += [decoded_input]
    if not calibrated:
        print(len(aggregated_input))
        if len(aggregated_input) < 900:
            continue
        else:
            aggregated_input_np = np.array(aggregated_input)
            skywalk_min = np.min(aggregated_input_np, axis=0)
            skywalk_max = np.max(aggregated_input_np, axis=0)
            calibrated = True
            aggregated_input = []

    if len(aggregated_input) != 90:
        continue
    aggregated_input_np = np.array(aggregated_input)
    for i in range(aggregated_input_np.shape[1]):
        aggregated_input_np[:, i] = normalize(aggregated_input_np[:, i], skywalk_min[i], skywalk_max[i])
    input_tensor = torch.FloatTensor([aggregated_input_np[[-1]]])
    output_tensor = mod(input_tensor)
    aggregated_input = aggregated_input[1:]
    print(f"{output_tensor[0, 0]: 0.3f} \t {output_tensor[0, 1]: 0.3f} \t {output_tensor[0, 2]: 0.3f} \t {output_tensor[0, 3]: 0.3f}")
