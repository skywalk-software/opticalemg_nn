import io
import math
import statistics
from typing import cast
import PIL
import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.transforms import ToTensor
from utils.postprocessing import plot_predictions
from utils.metrics import process_clicks


class ResidualBlock(nn.Module):
    def __init__(self, input_seq_length, input_channel, output_channel, middle_channel, dilation,
                 kernel_size):
        super().__init__()
        self.input_seq_length = input_seq_length
        self.output_seq_length = input_seq_length - (kernel_size - 1) * dilation
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.middle_channel = middle_channel
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv1d(input_channel, middle_channel, kernel_size, dilation=self.dilation)
        self.dropout1 = nn.Dropout(0.25)
        self.batchnorm1 = nn.BatchNorm1d(middle_channel)
        self.conv2 = nn.Conv1d(middle_channel, output_channel, 1)
        self.dropout2 = nn.Dropout(0.25)
        self.batchnorm2 = nn.BatchNorm1d(output_channel)

    def forward(self, x):
        residual = x[:, :, self.input_seq_length - self.output_seq_length: self.input_seq_length]
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x += residual
        return x


class SkywalkCnnV1(pl.LightningModule):
    def __init__(self, kernel_size: int, in_channels: int, seq_length: int, session_type='all'):
        super(SkywalkCnnV1, self).__init__()
        self.save_hyperparameters()
        self.session_type = session_type
        self.kernel_size = kernel_size
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.25),
        )
        self.layer2 = ResidualBlock(seq_length - (kernel_size - 1), 32, 32, 32, 3, kernel_size)
        self.layer3 = ResidualBlock(self.layer2.output_seq_length, 32, 32, 32, 9, kernel_size)
        self.layer4 = ResidualBlock(self.layer3.output_seq_length, 32, 32, 32, 27, kernel_size)
        self.layer5 = ResidualBlock(self.layer4.output_seq_length, 32, 32, 32, 81, kernel_size)
        self.layer6 = nn.Linear(self.layer5.output_seq_length * 32, 2)
        # for module in self.layers.modules():
        #     if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d):
        #         nn.init.xavier_uniform_(module.weight.data)
        #         # module.weight.data.fill_(1)
        #         module.bias.data.fill_(0)
        self.loss = nn.CrossEntropyLoss(reduction='none')
        self.hparams.lr = 0.001

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = torch.flatten(x, 1)
        x = self.layer6(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train/loss",
                "frequency": 1
            },
        }

    def training_step(self, batch, batch_idx):
        x, y, ts, meta = batch
        y_hat = self(x)
        raw_loss = self.loss(y_hat, y)
        loss = torch.sum(raw_loss * w) / torch.sum(w)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        x, y, w = batch
        y_hat = self(x)
        # y_same = torch.vstack([1 - y, y]).T
        # if batch_idx == 0 and dataset_idx == 0:
        #     tensorboard = cast(TensorBoardLogger, self.logger).experiment
        #     tensorboard.add_graph(self, x)
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