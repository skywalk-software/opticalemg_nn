import statistics, math
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch
import io
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import cast
import numpy as np
from torchvision.transforms import ToTensor
import PIL, matplotlib
from matplotlib import pyplot as plt
from pytorch_lightning.loggers import TensorBoardLogger
from utils.postprocessing import plot_predictions
from utils.metrics import process_clicks

def build_skywalkcnnv1(model_cfg, optimizer_cfg):
    pass

class ResidualBlock(nn.Module):
    def __init__(
        self, 
        input_len, 
        channels,
        dilation,
        stride,
        kernel_size,
        dropout,
    ):
        super().__init__()        
        self.input_len= input_len
        self.output_len = input_len - (kernel_size - 1) * dilation
        self.channels = channels
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.stride = stride
        self.dropout = dropout
        self.layers = nn.ModuleList([])
        for i in range(len(channels) - 1):
            self.layers.append(
                nn.Conv1d(
                    in_channels=self.channels[i], 
                    out_channels=self.channels[i + 1], 
                    kernel_size=kernel_size,
                    dilation=dilation,
                    stride=stride,
                ),
                nn.BatchNorm1d(self.channels[i + 1]),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            )
        self.network = nn.Sequential(*self.layers)

    def forward(self, x):
        residual = x[:, :, self.input_len - self.output_len: self.input_len]
        x = self.network(x)
        x += residual
        return x


class SkywalkCnnV1(pl.LightningModule):
    def __init__(
        self, 
        seq_length,
        kernel_size, 
        in_channels,
        channels,
        dilations,
        droput,
        val_dataset_names,
        test_dataset_names,
        optim_cfg, 
        session_type='all'
    ):
        super(SkywalkCnnV1, self).__init__()
        self.save_hyperparameters()

        self.val_dataset_names = val_dataset_names
        self.test_dataset_names = test_dataset_names
        self.session_type = session_type
        self.kernel_size = kernel_size
        self.channels = channels
        self.dilations = dilations
        self.dropout = droput

        self.layers = nn.ModuleList([])
        self.layers.append(
            nn.Sequential(
                nn.Conv1d(in_channels, self.channels[0], kernel_size),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(0.25),
            )
        )
        input_len = seq_length - (kernel_size - 1)
        for i in range(len(channels) - 1):
            layer = ResidualBlock(
                input_len=input_len,
                channels=channels,
                dilation=dilations[i],
                stride=1,
                kernel_size=kernel_size,
            )
            input_len = layer.output_len
            self.layers.append(layer)

        self.network = nn.Sequential(
            *self.layers,
            nn.Flatten(),
            nn.Linear(input_len * channels[-1], 2),
        )

        self.loss = nn.BCEWithLogitsLoss(reduction='none')
        self.optim_cfg = optim_cfg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.transpose(x, 1, 2)
        x = self.network(x)
        return x

    def configure_optimizers(self):
        cfg = self.optim_cfg
        if cfg.type == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=cfg.lr)
        if cfg.scheduler == 'reduce_lr':
            scheduler = ReduceLROnPlateau(optimizer, 'min', factor=cfg.factor)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train/loss",
                "frequency": 1
            },
        }

    def training_step(self, batch, batch_idx):
        x, y, w = batch
        y_hat = self(x)
        raw_loss = self.loss(y_hat, y)
        loss = torch.sum(raw_loss * w) / torch.sum(w)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, loader_idx):
        x, y, w = batch
        y_hat = self(x)
        return y.cpu(), y_hat.detach().cpu(), w.cpu()

    def validation_epoch_end(self, outputs):
        tensorboard = cast(TensorBoardLogger, self.logger).experiment

        val_prefix = f"val-{self.session_type}/"
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

        if not isinstance(outputs[0], list):
            outputs = [outputs]

        for dataset_idx, dataset_outputs in enumerate(outputs):
            dataset_prefix = f"val-full/{self.val_dataset_names[dataset_idx]}/"
            y_list = []
            y_hat_list = []
            w_list = []
            for batch_idx, (y, y_hat, w) in enumerate(dataset_outputs):
                y_list += [y]
                y_hat_list += [y_hat]
                w_list += [w]
            total_y = torch.cat(y_list)
            total_y_hat = torch.cat(y_hat_list)
            total_w = torch.cat(w_list)

            assert len(total_y) == len(total_y_hat)

            samples = len(total_y)

            raw_loss = self.loss(total_y_hat, total_y)
            loss = torch.sum(total_w * raw_loss) / torch.sum(total_w)
            estimated_y = (total_y_hat[:, 0] < total_y_hat[:, 1]).long()
            accuracy = torch.sum((estimated_y == total_y)) / samples

            result = process_clicks(total_y, estimated_y)
            total_true_clicks, total_false_clicks, total_missed_clicks, total_detected_clicks, \
            on_set_offsets, off_set_offsets, drops = result
            assert len(on_set_offsets) == len(off_set_offsets) == total_detected_clicks
            self.log(f"{dataset_prefix}loss", loss)
            self.log(f"{dataset_prefix}accuracy", accuracy)
            self.log(f"{dataset_prefix}FP-P",
                     math.nan if total_true_clicks == 0 else total_false_clicks / total_true_clicks)
            self.log(f"{dataset_prefix}TP-P",
                     math.nan if total_true_clicks == 0 else total_detected_clicks / total_true_clicks)
            self.log(f"{dataset_prefix}drops-P", math.nan if total_true_clicks == 0 else len(drops) / total_true_clicks)
            self.log(f"{dataset_prefix}std-onset",
                     math.nan if total_detected_clicks < 2 else statistics.stdev(on_set_offsets))
            self.log(f"{dataset_prefix}std-offset",
                     math.nan if total_detected_clicks < 2 else statistics.stdev(off_set_offsets))

            if total_detected_clicks > 1:
                tensorboard.add_histogram(f"{dataset_prefix}onset", np.array(on_set_offsets), self.current_epoch)
                tensorboard.add_histogram(f"{dataset_prefix}offset", np.array(off_set_offsets), self.current_epoch)

            if len(drops) > 1:
                tensorboard.add_histogram(f"{dataset_prefix}drops", np.array(drops), self.current_epoch)

            length = len(estimated_y)
            # if length < 1000:
            #     start = 0
            #     end = length
            # else:
            #     start = length // 2 - 500
            #     end = length // 2 + 500
            start = 0
            end = length
            fig: matplotlib.figure = plot_predictions(estimated_y[start:end].numpy(), total_y[start:end].numpy(), None)
            # fig: matplotlib.figure = plot_predictions(estimated_y.numpy(), total_y.numpy(), None)
            # fig.show()

            buf = io.BytesIO()
            fig.set_size_inches((length // 100, 3))
            fig.savefig(buf, format='jpeg')
            buf.seek(0)
            image = PIL.Image.open(buf)
            image = ToTensor()(image)
            plt.close(fig)
            tensorboard.add_image(f"{dataset_prefix}predictions", image, self.current_epoch)

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
        fp_p = math.nan if val_total_true_clicks == 0 else val_total_false_clicks / val_total_true_clicks
        self.log(f"{val_prefix}FP-P", fp_p)
        tp_p = math.nan if val_total_true_clicks == 0 else val_total_detected_clicks / val_total_true_clicks
        self.log(f"{val_prefix}TP-P", tp_p)
        drops_p = math.nan if val_total_true_clicks == 0 else len(val_drops) / val_total_true_clicks
        self.log(f"{val_prefix}drops-P", drops_p)
        std_onset = math.nan if val_total_detected_clicks < 2 else statistics.stdev(val_on_set_offsets)
        self.log(f"{val_prefix}std-onset", std_onset)
        std_offset = math.nan if val_total_detected_clicks < 2 else statistics.stdev(val_off_set_offsets)
        self.log(f"{val_prefix}std-offset", std_offset)

        print(f"[for notion reporting] + {self.session_type}"
              f"\t{tp_p:.4f}\t{fp_p:.4f}\t{drops_p:.4f}\t{val_loss:.4f}\t{std_onset:.4f}\t{std_offset:.4f}")

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

