'''Module to train&test model of choise'''
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import torch
import yaml
import wandb


from DataModules import dataset_selector
from Models import resnet19
from LightningModules import lightning_model


def main() -> None:
    '''Function to test and train a model'''
    # Load config file
    with open('Python/Lightning_Resnet19/config.yaml', 'r', encoding="utf-8") as file:
        config = yaml.safe_load(file)

    pl.seed_everything(1)
    wandb.login()
    torch.set_float32_matmul_precision("medium")

    # Data Module
    #dm = dataset_selector.CIFAR10DataModule(batch_size=config["batch_size"])
    img_path = 'Python/Lightning_Resnet19/DataModules/datasets/hbku2019/imgs/train'
    csv_path = 'Python/Lightning_Resnet19/DataModules/datasets/hbku2019/labels/labels_train.csv'
    dm = dataset_selector.Hbku2019DataModule(img_path, csv_path)
    # Model
    #model_name = 'snn'
    models = ['conv']#'fc','conv','snnDrop','snn'
    num_classes = dm.classes
    for model_name in models:
        model = resnet19.select_model(model_name, num_classes)

        # Trainer
        callbacks=[ModelCheckpoint(
            dirpath="Python/Lightning_Resnet19/checkpoints",
            every_n_epochs=2,
            filename= model_name,
            auto_insert_metric_name = True,
            save_top_k=1,
            monitor="val/mAP",
            mode="max",
        ),]
        # Lightning Module
        lit_model = lightning_model.PlModuleCreator(model, config, num_classes)
        # Logger
        wandb_logger = WandbLogger(
            project="multilabel_resnet19", name=model_name, log_model="all")
        trainer = pl.Trainer(
            max_epochs=config['max_epochs'], logger=wandb_logger, callbacks=callbacks)
        trainer.fit(lit_model, datamodule=dm)
        trainer.test(lit_model, datamodule=dm)
        wandb.finish()


if __name__ == '__main__':
    main()
