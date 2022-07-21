import hydra
import logging
from omegaconf import OmegaConf
from models.skywalk_model import SkywalkCnnV1
import os
from utils.dataloader import get_dataloaders
from torchsummary import summary
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
import sys

@hydra.main(version_base="1.2.0", config_path="./config", config_name="conf")
def main(cfg):

    logger = logging.getLogger(__name__)        

    trainloader, valloader, testloader = get_dataloaders(cfg.data, cfg.dataset, cfg.dataloader)

    samp = trainloader.dataset[0]
    nir, _, _, _, _, _ = samp
    n_channels = nir.shape[0]
    seq_len = cfg.dataset.seq_length

    logger.info(f"In data channels: {n_channels}")
    logger.info(f"Model training on sequence length: {seq_len}")

    model = SkywalkCnnV1(
        kernel_size=cfg.model.kernel_size, 
        seq_length=seq_len,
        in_channels=n_channels,
    )

    summary(model, nir.shape, device='cpu')

    CKPT_PATH = f"{os.getcwd()}/ckpts/model.ckpt"

    tboard_logger = TensorBoardLogger(name="logs", save_dir=f"{os.getcwd()}/logs")

    trainer = Trainer(
        accelerator="cpu" if sys.platform == 'darwin' else "auto",  # temp fix for mps not working
        max_epochs=cfg.epochs,
        logger=tboard_logger,
        val_check_interval=1.0,
        callbacks=[
            LearningRateMonitor(logging_interval='epoch')
        ],
        enable_checkpointing=False
    )

    trainer.fit(model, train_dataloaders=trainloader, val_dataloaders=valloader)
    trainer.save_checkpoint(CKPT_PATH)

    trainer.test(model, test_dataloaders=testloader)

if __name__ == '__main__':
    main()