import hydra
import logging
from omegaconf import OmegaConf
from models.skywalk_model import SkywalkCnnV1
import pytorch_lightning as pl 
from utils.dataloader import get_dataloaders


@hydra.main(version_base="1.2.0", config_path="./config", config_name="conf")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))

    logger = logging.getLogger(__name__)        

    logger.info("test info message!!")

    trainloader, valloader, testloader = get_dataloaders(cfg.data, cfg.dataset, cfg.dataloader)

    samp = trainloader.dataset[0]
    nir, _, _, _ = samp
    n_channels = nir.shape[0]
    seq_len = cfg.dataset.seq_length

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

    kernel_size = 3
    epochs = 5
    next_epochs = 20

    summary(model, nir.shape, device='cpu')

    CKPT_PATH = "./saved_model.ckpt"

    # %% training
    logger = TensorBoardLogger('logs')

    trainer = Trainer(
        accelerator="cpu" if sys.platform == 'darwin' else "auto",  # temp fix for mps not working
        max_epochs=epochs,
        logger=logger,
        val_check_interval=1.0,
        callbacks=[
            LearningRateMonitor(logging_interval='epoch')
        ]
    )

    trainer.fit(model, train_dataloaders=trainloader, val_dataloaders=valloader)
    trainer.save_checkpoint(CKPT_PATH)

    # # %% Test the model metrics
    # trainer = Trainer(
    #     resume_from_checkpoint=CKPT_PATH
    # )
    # model.load_from_checkpoint(CKPT_PATH)
    # print("test result:")
    # print(trainer.validate(model, dataloaders=testloader))

    # trainer = Trainer(
    #     resume_from_checkpoint=CKPT_PATH
    # )
    # model.load_from_checkpoint(CKPT_PATH)


    # dataloader = valloader[0]
    # dataloader_name = val_sessions_meta_names[0]
    # print(f"plotting on {dataloader_name}")
    # # device = torch.device("cpu" if sys.platform == 'darwin' else "cuda")
    # device = torch.device("cpu")

    # model_device = model.to(device)
    
    # y_all = []
    # y_hat_all = []
    # model_device.eval()
    # for x, y, w in tqdm.tqdm(dataloader):
    #     y_hat_float = model_device(x.to(device)).detach().cpu()
    #     y_hat = (y_hat_float[:, 0] < y_hat_float[:, 1]).long()
    #     y_hat_all += [y_hat]
    #     y_all += [y]

    # y_all_np = torch.cat(y_all).numpy()
    # y_hat_all_np = torch.cat(y_hat_all).numpy()
    # # dirty hack to retrieve data from dataloader
    # x_all_np = cast(SkywalkDataset, dataloader.dataset).data_array[:len(y_all_np)].numpy()

    # fig = plot_predictions(y_hat_all_np, y_all_np, x_all_np)
    # fig.show()




if __name__ == '__main__':
    main()