import hydra
import logging
from omegaconf import OmegaConf
from models.cnnv1 import SkywalkCnnV1
import pytorch_lightning as pl 
from utils.dataloader import get_dataloaders


@hydra.main(version_base="1.2.0", config_path="./config", config_name="conf")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))

    logger = logging.getLogger(__name__)        

    logger.info("test info message!!")

    trainloader, valloader, testloader = get_dataloaders(cfg.data, cfg.dataset)

    model = SkywalkCnnV1(cfg.model)

    callbacks = []

    if cfg.early_stopping:
        # early_stopping = pl.callbacks.EarlyStopping()
        pass

    if cfg.enable_checkpointing:
        checkpointing = pl.callbacks.ModelCheckpoint(
            dirpath=cfg.checkpoint_dir,
            filename="{epoch}.ckpt",
            save_last=True,
            monitor=cfg.checkpoint_monitor,
            save_top_k=cfg.checkpoint_top_k,
        )
        callbacks.append(checkpointing)

    trainer = pl.Trainer(
        accelerator="cpu",
        deterministic=cfg.deterministic,
        fast_dev_run=cfg.dev_run,
        enable_progress_bar=cfg.progress_bar,
        resume_from_checkpoint=cfg.resume_training,
        enable_model_summary=cfg.print_summary,
        detect_anomaly=cfg.detect_anomaly,
        check_val_every_n_epoch=cfg.val_freq,
        callbacks=callbacks
    )

    trainer.fit(
        model=model,
        train_dataloaders=trainloader,
        val_dataloaders=valloader,
    )





if __name__ == '__main__':
    main()