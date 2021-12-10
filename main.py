import logging

import numpy as np
import pytorch_lightning as pl
import torch.optim
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.utils.data import DataLoader, Dataset, ConcatDataset

from dataloader import load_h5
from model import NnModel

# %%

all_data_path = [
    # "../dataset/data5/2021-12-01T16-44-19.h5",
    # "../dataset/data5/2021-12-01T23-21-29.h5",
    # "../dataset/data5/2021-12-01T23-52-03.h5",
    # "../dataset/data5/2021-12-02T00-22-55.h5",
    # "../dataset/data5/2021-12-02T10-48-48.h5",
    # "../dataset/data5/2021-12-02T16-39-06.h5",
    "../dataset/data5/2021-12-02T17-24-05.h5",
]
all_data = list(zip(*[load_h5(path) for path in all_data_path]))

train_dataset, val_dataset = ConcatDataset(all_data[0]), ConcatDataset(all_data[1])

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

mod = NnModel(learning_rate=0.0005, gamma=1)
# trainer = pl.Trainer(max_epochs=6, callbacks=[checkpoint_callback])
lr_monitor = LearningRateMonitor(logging_interval='epoch')
trainer = pl.Trainer(max_epochs=30, callbacks=[lr_monitor])

# %%
# mod.load_from_checkpoint("nn.ckpt", learning_rate=0.001)
# trainer = pl.Trainer(resume_from_checkpoint="nn.ckpt", max_epochs=101)
trainer.tune(mod, datamodule=SkywalkDataModel())

# %%
trainer.fit(model=mod, datamodule=SkywalkDataModel())

# %%
trainer.save_checkpoint("../models/nn2_5_1.ckpt")

# %%
loss_arr = []
for x, y in reversed(val_dataset):
    x, y = torch.Tensor([x]), torch.Tensor([y])
    mod.log = lambda x, y: print(x, y)
    loss_arr += [mod.validation_step((x, y), 0)]

# %%

np.set_printoptions(precision=4, suppress=True)

ref_a = []
ref_b = []
ref_x = []
ref_y = []
out_x = []
out_y = []
out_a = []
out_b = []
# x_indexes = list(range(63))
x_indexes = [7, 8]
test_x = [[] for x in x_indexes]
mod.eval()

for data_x, data_y in reversed(val_dataset):
    ans_y = mod(torch.Tensor([data_x]))
    out_a += [np.array(ans_y.detach().numpy()[0])[0]]
    out_b += [np.array(ans_y.detach().numpy()[0])[1]]
    out_x += [np.array(ans_y.detach().numpy()[0])[2]]
    out_y += [np.array(ans_y.detach().numpy()[0])[3]]
    ref_a += [data_y[0][1:][0]]
    ref_b += [data_y[0][1:][1]]
    ref_x += [data_y[0][1:][2]]
    ref_y += [data_y[0][1:][3]]
    for i, x_idx in enumerate(x_indexes):
        test_x[i] += [data_x[0][x_idx]]
    print("out", np.array(ans_y.detach().numpy()[0]))
    print("ans", data_y[0][1:])