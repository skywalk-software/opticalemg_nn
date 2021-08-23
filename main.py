import logging
import math

import h5py
import pytorch_lightning as pl
import torch.optim
from pytorch_lightning.callbacks import LearningRateMonitor
from torch import nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import numpy as np

from skywalk_leap_dataset import SkywalkLeapDataset
import matplotlib

matplotlib.use('TkAgg')


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
    return (arr - min_val) / (max_val - min_val + 1e-7)


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
        skywalk_data[:, i] = skywalk_data[:, i] / 10000

    leapmotion_data = np.array(leapmotion_data)
    # leapmotion_data[:, 1] = (leapmotion_data[:, 1] > - math.pi / 4) * leapmotion_data[:, 1] - (leapmotion_data[:, 1] <= - math.pi / 4) * leapmotion_data[:, 1]
    # leapmotion_data[:, 3] = - leapmotion_data[:, 1]
    # leapmotion_data[:, 4] = - leapmotion_data[:, 2]
    # leapmotion_data[:, 1:5][leapmotion_data[:, 1:5] < 0] = 0
    # leapmotion_data[:, 3] = normalize(leapmotion_data[:, 3])
    # leapmotion_data[:, 4] = normalize(leapmotion_data[:, 4])
    # leapmotion_data[:, 3] = leapmotion_data[:, 3]
    # leapmotion_data[:, 4] = leapmotion_data[:, 4]
    print("dataset_length", len(skywalk_timestamps))

    train_val_div = len(skywalk_timestamps) // 4 * 3
    train_dataset = SkywalkLeapDataset(
        1,
        leapmotion_timestamps,
        leapmotion_data,
        skywalk_timestamps[200:train_val_div],
        skywalk_data[200:train_val_div]
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
    # "data4/2021-05-16T15-56-44.h5"
    # "/Users/jackie/Documents/proc/proto2_data1/2021-07-26T22-01-22.h5"
    # "/Users/jackie/Documents/proc/proto2_data1/2021-08-09T16-57-51.h5"
    # "/Users/jackie/Documents/proc/proto2_data1/2021-08-11T20-06-18.h5"
    "/Users/jackie/Documents/proc/proto2_data1/2021-08-18T20-15-53.h5",
    "/Users/jackie/Documents/proc/proto2_data1/2021-08-18T20-31-46.h5"
]
all_data = list(zip(*[load_h5(path) for path in all_data_path]))

train_dataset, val_dataset = ConcatDataset(all_data[0]), ConcatDataset(all_data[1])

# %%
# sw_idx, lm_idx = np.array(list(zip(*all_data[0][0].skywalk_idx_to_leapmotion_map)))
#
# lm = normalize(np.sum(np.abs(np.diff(all_data[0][0].leapmotion_data[lm_idx], axis=0)), axis=1))
# sw = normalize(np.sum(np.abs(np.diff(all_data[0][0].skywalk_data[sw_idx], axis=0)), axis=1))
#
# # %%
#
# import matplotlib.pyplot as plt
#
# start = 100
# range_len = 500
# end = start + range_len
#
# plt.plot(range(range_len), lm[start:end])
# plt.show()
#
# # %%
#
# plt.plot(range(range_len), sw[start:end])
# plt.show()
#
# # %%
# lm_labels = ["event.valid", "event.pitch", "event.yaw", "event.thumbDist", "event.otherDist"]
# # for i in [0, 1, 2, 3, 4]:
# for i in [1, 2]:
#     plt.plot(range(range_len), normalize(all_data[0][0].leapmotion_data[lm_idx][start:end][:, i]), label=lm_labels[i])
# plt.legend()
# plt.show()
#
# # %%
# for i in range(62):
#     plt.plot(range(range_len), normalize(all_data[0][0].skywalk_data[sw_idx][start:end][:, i]), label=f"channel {i}")
# plt.legend()
# plt.ylim(-1, 2)
# plt.show()
#
# # %%
# start = 0
# range_len = 2000
# end = start + range_len
#
# input_group = []
# output_group = []
# for idx in range(start, end):
#     input_group += [train_dataset[idx][0][0]]
#     output_group += [train_dataset[idx][1][0]]
# input_np = np.array(input_group)
# output_np = np.array(output_group)
#
# lm_labels = ["event.valid", "event.pitch", "event.yaw", "event.thumbDist", "event.otherDist"]
# for i in range(5):
#     plt.plot(range(range_len), output_np[:, i], label=lm_labels[i])
# plt.legend()
# plt.show()
#
# for i in range(12):
#     plt.plot(range(range_len), input_np[:, i], label=f"channel {i}")
# plt.ylim((-0.5, 1.5))
# plt.legend()
# plt.show()


# %%

class NnModel(pl.LightningModule):
    def __init__(self, learning_rate=0.001, gamma=0.95):
        super().__init__()
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.mlp_1 = nn.Sequential(
            nn.Linear(in_features=64, out_features=48),
            nn.ReLU()
        )
        self.mlp_2 = nn.Sequential(
            nn.Linear(in_features=48, out_features=32),
            nn.ReLU()
        )
        self.mlp_3 = nn.Sequential(
            nn.Linear(in_features=32, out_features=24),
            nn.ReLU()
        )
        self.mlp_4 = nn.Sequential(
            nn.Linear(in_features=24, out_features=12),
            nn.ReLU()
        )
        self.mlp_5 = nn.Sequential(
            nn.Linear(in_features=12, out_features=4)
        )
        self.loss = nn.MSELoss(reduction='none')

    def forward(self, x):
        x = x[:, -1, :]
        out = self.mlp_5(self.mlp_4(self.mlp_3(self.mlp_2(self.mlp_1(x)))))
        return out

    def loss_fn(self, out, target):
        return self.loss(out, target)

    def configure_optimizers(
            self,
    ):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        # return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val/loss"}
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val/loss"}

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        out = out.unsqueeze(1).repeat(1, y.shape[1], 1)
        loss = torch.mean(self.loss_fn(out, y[:, :, 1:]), dim=0)
        pitch_loss, yaw_loss, thumb_dist_loss, other_dist_loss = torch.min(loss, dim=0)[0]
        total_loss = pitch_loss + yaw_loss + thumb_dist_loss / 1000 + other_dist_loss / 10000
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
        total_loss = pitch_loss + yaw_loss + thumb_dist_loss / 1000 + other_dist_loss / 10000
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
logging.getLogger().setLevel(logging.DEBUG)

# checkpoint_callback = pl.callbacks.ModelCheckpoint(
#     monitor='valid_loss',
#     dirpath='./',
#     filename='models-{epoch:02d}-{valid_loss:.2f}',
#     save_top_k=3,
#     mode='min')

mod = NnModel(learning_rate=0.0005, gamma=1)
# trainer = pl.Trainer(max_epochs=6, callbacks=[checkpoint_callback])
lr_monitor = LearningRateMonitor(logging_interval='epoch')
trainer = pl.Trainer(max_epochs=100, callbacks=[lr_monitor])

# %%
# mod.load_from_checkpoint("nn.ckpt", learning_rate=0.001)
# trainer = pl.Trainer(resume_from_checkpoint="nn.ckpt", max_epochs=101)
# trainer.tune(mod, datamodule=SkywalkDataModel())

#%%
trainer.fit(model=mod, datamodule=SkywalkDataModel())

# %%
# trainer.save_checkpoint("nn2_1.ckpt")

# #%%
mod = NnModel.load_from_checkpoint("nn2_1.ckpt", learning_rate=0.001)

# %%
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)

ref_x = []
ref_y = []
out_x = []
out_y = []
# x_indexes = list(range(63))
x_indexes = [7, 8]
test_x = [[] for x in x_indexes]
test_range = list(range(5000))

for idx in test_range:
    data_x, data_y = train_dataset[idx]
    ans_y = mod(torch.Tensor([data_x]))
    out_x += [np.array(ans_y.detach().numpy()[0])[0]]
    out_y += [np.array(ans_y.detach().numpy()[0])[1]]
    ref_x += [data_y[0][1:][0]]
    ref_y += [data_y[0][1:][1]]
    for i, x_idx in enumerate(x_indexes):
        test_x[i] += [data_x[0][x_idx]]
    print("out", np.array(ans_y.detach().numpy()[0]))
    print("ans", data_y[0][1:])

plt.clf()
plt.plot(test_range, ref_x, label="ref_x")
plt.plot(test_range, ref_y, label="ref_y")
plt.plot(test_range, out_x, label="out_x")
plt.plot(test_range, out_y, label="out_y")
# for i, x_idx in enumerate(x_indexes):
#     plt.plot(test_range, normalize(test_x[i]), label=f"test_x_{x_idx}")
plt.legend()
plt.ylim(-2, 2)

# #%%
# import csv
#
# with open('data.csv', 'w') as csvfile:
#     datawriter = csv.writer(csvfile)
#     datawriter.writerow(["ref_x", "ref_y", "out_x", "out_y"] + [f"test_x_{x_idx}" for x_idx in x_indexes])
#     n_test_x = [normalize(test_x[i]) for i in range(len(x_indexes))]
#     for i, _ in enumerate(test_range):
#         datawriter.writerow([ref_x[i], ref_y[i], out_x[i], out_y[i]] + [n_test_x[j][i] for j in range(len(x_indexes))])


# %%
# while True:
#     line = skywalk_serial.readline()
#     decoded_input = [float(item) for item in line.decode('utf-8').split(",")[:-1]]
#     if len(decoded_input) != 12:
#         print(decoded_input)
#         continue
#     aggregated_input += [decoded_input]
#     if not calibrated:
#         print(len(aggregated_input))
#         if len(aggregated_input) < 900:
#             continue
#         else:
#             aggregated_input_np = np.array(aggregated_input)
#             skywalk_min = np.min(aggregated_input_np, axis=0)
#             skywalk_max = np.max(aggregated_input_np, axis=0)
#             calibrated = True
#             aggregated_input = []
#
#     if len(aggregated_input) != 90:
#         continue
#     aggregated_input_np = np.array(aggregated_input)
#     for i in range(aggregated_input_np.shape[1]):
#         aggregated_input_np[:, i] = normalize(aggregated_input_np[:, i], skywalk_min[i], skywalk_max[i])
#     input_tensor = torch.FloatTensor([aggregated_input_np[[-1]]])
#     output_tensor = mod(input_tensor)
#     aggregated_input = aggregated_input[1:]
#     print(f"{output_tensor[0, 0]: 0.3f} \t {output_tensor[0, 1]: 0.3f} \t {output_tensor[0, 2]: 0.3f} \t {output_tensor[0, 3]: 0.3f}")

# %%

import numpy as np
import serial
import multiprocess
from multiprocess import Process, Value, Array, Lock

calibrated = True
skywalk_min = None
skywalk_max = None

aggregated_input = []

multiprocess.set_start_method('spawn')


# %%
arr = Array("d", [.0, .0, .0, .0])
l = Lock()


def anim_process(l, arr):
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import numpy as np
    data = [0] * 30, [0] * 30, [0] * 30, [0] * 30
    fig, ax = plt.subplots()
    rects = plt.bar(range(4), np.random.rand(4))

    def run(new_data):
        del data[0][0]
        del data[1][0]
        del data[2][0]
        del data[3][0]
        data[0].append(new_data[0])
        data[1].append(new_data[1])
        data[2].append(new_data[2])
        data[3].append(new_data[3])
        ax.set_ylim(-2, 2)
        for rect, new_data_point in zip(rects, new_data):
            rect.set_height(new_data_point)
        return rects

    def data_gen():
        t = 0
        while True:
            with l:
                # print(arr[0], arr[1], arr[2], arr[3])
                yield arr[0], arr[1], arr[2], arr[3]
        print("exited!")

    ani = animation.FuncAnimation(fig, run, frames=data_gen, interval=1, blit=False)
    plt.show()

# anim_process()

p = Process(target=anim_process, daemon=True, args=(l, arr, ))

p.start()

# %%

SERIAL_PORT = "/dev/cu.usbmodem88100401"
SERIAL_BAUD_RATE = 115200
skywalk_serial = serial.Serial(port=SERIAL_PORT, baudrate=SERIAL_BAUD_RATE)
# read until timeout
skywalk_serial.timeout = 0
while True:
    try:
        line = skywalk_serial.readline()
        print(line)
        if line == b'':
            break
    except serial.SerialTimeoutException:
        break
# resume unlimited timeout
skywalk_serial.timeout = None

# process skywalk data
while True:
    line = skywalk_serial.readline()
    decoded_input = [float(item) / 10000 for item in line.decode('utf-8').split(",")[:-1]]
    if len(decoded_input) != 64:
        print(decoded_input)
        continue
    # global aggregated_input, calibrated, skywalk_min, skywalk_max
    # aggregated_input += [decoded_input]
    # if not calibrated:
    #     print(len(aggregated_input))
    #     if len(aggregated_input) < 300:
    #         continue
    #     else:
    #         aggregated_input_np = np.array(aggregated_input)
    #         skywalk_min = np.min(aggregated_input_np, axis=0)
    #         skywalk_max = np.max(aggregated_input_np, axis=0)
    #         calibrated = True
    #         aggregated_input = []

    # if len(aggregated_input) != 90:
    #     continue
    aggregated_input_np = np.array(aggregated_input)
    # for i in range(aggregated_input_np.shape[1]):
    #     aggregated_input_np[:, i] = normalize(aggregated_input_np[:, i], skywalk_min[i], skywalk_max[i])
    input_tensor = torch.FloatTensor([[decoded_input]])
    output_tensor = mod(input_tensor)
    print("input_tensor", input_tensor)
    print("output_tensor", output_tensor)
    aggregated_input = aggregated_input[1:]
    print(f"{output_tensor[0, 0]: 0.3f} \t {output_tensor[0, 1]: 0.3f}")
    with l:
        arr[0] = output_tensor[0, 0].item()
        arr[1] = output_tensor[0, 1].item()
        arr[2] = output_tensor[0, 2].item()
        arr[3] = output_tensor[0, 3].item()
