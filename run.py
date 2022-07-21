import hydra
import logging
from omegaconf import OmegaConf
from models.skywalk_model import SkywalkCnnV1
import os
from utils.dataloader import get_dataloaders

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

    # callbacks = []

    # if cfg.early_stopping:
        # early_stopping = pl.callbacks.EarlyStopping()
    #     pass

    # if cfg.enable_checkpointing:
    #     checkpointing = pl.callbacks.ModelCheckpoint(
    #         dirpath=cfg.checkpoint_dir,
    #         filename="{epoch}.ckpt",
    #         save_last=True,
    #         monitor=cfg.checkpoint_monitor,
    #         save_top_k=cfg.checkpoint_top_k,
    #     )
    #     callbacks.append(checkpointing)

    # trainer = pl.Trainer(
    #     accelerator="cpu",
    #     deterministic=cfg.deterministic,
    #     fast_dev_run=cfg.dev_run,
    #     enable_progress_bar=cfg.progress_bar,
    #     enable_model_summary=cfg.print_summary,
    #     detect_anomaly=cfg.detect_anomaly,
    #     check_val_every_n_epoch=cfg.val_freq,
    #     callbacks=callbacks,
    #     max_epochs=cfg.epochs,
    # )

    # trainer.fit(
    #     model=model,
    #     train_dataloaders=trainloader,
    #     val_dataloaders=valloader,
    # )

    from torchsummary import summary
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import LearningRateMonitor
    import sys

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