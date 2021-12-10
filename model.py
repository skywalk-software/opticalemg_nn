from abc import ABC

import pytorch_lightning as pl
import torch
from torch import nn


class NnModel(pl.LightningModule, ABC):
    def __init__(self, learning_rate=0.001, gamma=0.95):
        super().__init__()
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.mlp_1 = nn.Sequential(
            nn.Conv1d(20, 32, kernel_size=3, padding=3, stride=3),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.mlp_2 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=3, padding=3, stride=3),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.mlp_3 = nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=3, padding=3, stride=3),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.mlp_4 = nn.Sequential(
            nn.Conv1d(16, 8, kernel_size=3, padding=3, stride=3),
            nn.BatchNorm1d(8),
            nn.ReLU()
        )
        self.mlp_5 = nn.Sequential(
            nn.Conv1d(8, 4, kernel_size=3, padding=3, stride=3)
        )
        self.loss = nn.MSELoss(reduction='none')
        self.val_loss = nn.L1Loss(reduction='none')

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = x[:, :, [-1]].repeat(1, 1, 243)
        out = self.mlp_5(self.mlp_4(self.mlp_3(self.mlp_2(self.mlp_1(x)))))
        return torch.mean(out, dim=2)

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
        loss = torch.mean(self.val_loss(out, y[:, :, 1:]), dim=0)
        pitch_loss, yaw_loss, thumb_dist_loss, other_dist_loss = torch.min(loss, dim=0)[0]
        total_loss = pitch_loss + yaw_loss + thumb_dist_loss + other_dist_loss
        self.log("val/loss", total_loss)
        self.log("val/pitch_loss", pitch_loss)
        self.log("val/yaw_loss", yaw_loss)
        self.log("val/thumb_dist_loss", thumb_dist_loss)
        self.log("val/other_dist_loss", other_dist_loss)
        return total_loss
