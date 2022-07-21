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
        super(ResidualBlock, self).__init__()
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
        x, y, w, x_ts, y_ts, meta = batch
        y_hat = self(x)
        raw_loss = self.loss(y_hat, y.squeeze())
        loss = torch.sum(raw_loss * w.squeeze()) / torch.sum(w)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        x, y, w, x_ts, y_ts, meta = batch
        y_hat = self(x)
        return y.cpu(), y_hat.detach().cpu(), w.cpu(), meta

    def validation_epoch_end(self, outputs):
        tensorboard = cast(TensorBoardLogger, self.logger).experiment

        trial_types = []
        for _, _, _, meta in outputs:
            trial_types += meta
        trial_types = list(set(trial_types))

        results_dict = {}
        for trial_type in trial_types + ["total"]:
            results_dict[trial_type] = {
                'y_list': [],
                'y_hat_list': [],
                'w_list': [],
                'true_clicks': None,
                'false_clicks': None,
                'missed_clicks': None,
                'detected_clicks': None,
                'on_set_offsets': None,
                'off_set_offsets': None,
                'drops': None,
                'loss': None,
                'accuracy': None,
            }

        for batch in outputs:
            y, y_hat, w, m = batch
            for i in range(len(y)):
                results_dict[m[i]]['y_list'].append(y[i])
                results_dict[m[i]]['y_hat_list'].append(y_hat[i])
                results_dict[m[i]]['w_list'].append(w[i])
                results_dict["total"]['y_list'].append(y[i])
                results_dict["total"]['y_hat_list'].append(y_hat[i])
                results_dict["total"]['w_list'].append(w[i])

        to_log = ['accuracy', 'loss', 'FP-P', 'TP-P', 'drops-P', 'std-onset', 'std-offset']

        for k, v in results_dict.items():
            y = torch.cat(v['y_list'])
            y_hat = torch.stack(v['y_hat_list'])
            w = torch.cat(v['w_list'])

            assert y.shape[0] == y_hat.shape[0]
            assert y.shape[0] == w.shape[0]
            n_samples = y.shape[0]
            raw_loss = self.loss(y_hat, y)
            loss = torch.sum(w * raw_loss) / torch.sum(w)
            v['loss'] = loss.item()
            estimated_y = (y_hat[:, 0] < y_hat[:, 1]).long()
            accuracy = torch.sum((estimated_y == y)) / n_samples
            v['accuracy'] = accuracy.item()
            result = process_clicks(y, estimated_y)
            assert len(result['off_set_offsets']) == len(result['on_set_offsets']) == result['detected_clicks']
            for _k, _v in result.items():
                v[_k] = _v

            v['FP-P'] = math.nan if v['true_clicks'] == 0 else v['false_clicks'] / v['true_clicks']
            v['TP-P'] = math.nan if v['true_clicks'] == 0 else v['detected_clicks'] / v['true_clicks']
            v['drops-P'] = math.nan if v['true_clicks'] == 0 else len(v['drops']) / v['true_clicks']
            v['std-onset'] = math.nan if v['detected_clicks'] < 2 else np.std(v['on_set_offsets'])
            v['std-offset'] = math.nan if v['detected_clicks'] < 2 else np.std(v['off_set_offsets'])
            
            for _k, _v in v.items():
                if _k in to_log:
                    self.log(f"val/{k}/{_k}", _v)

            if v['detected_clicks'] > 1:
                tensorboard.add_histogram(f"val/{k}/onset", np.array(v['on_set_offsets']), self.current_epoch)
                tensorboard.add_histogram(f"{k}/offset", np.array(v['off_set_offsets']), self.current_epoch)

            if len(v['drops']) > 1:
                tensorboard.add_histogram(f"val/{k}/drops", np.array(v['drops']), self.current_epoch)



def get_predictions(model, test_dataset_for_pred):
    output_list = []
    for x, y in test_dataset_for_pred:
        output = model(torch.Tensor(x)).detach().numpy()
        output_list += [output]
    output_array = np.concatenate(output_list, axis=0)
    return np.argmax(output_array, axis=1)