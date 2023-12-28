'''Module to train&test model of choise'''
import yaml
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from DataModules import dataset_selector
from Models import resnet19
from LightningModules import lightning_model
import wandb


def main() -> None:
    '''Function to test and train a model'''
    # Load config file
    with open('Python/Lightning_Resnet19/config.yaml', 'r', encoding="utf-8") as file:
        config = yaml.safe_load(file)

    pl.seed_everything(1)
    wandb.login()

    # Data Module
    # dm = dataset_selector.CIFAR10DataModule(batch_size=config["batch_size"])
    img_path = 'Python/Lightning_Resnet19/DataModules/datasets/hbku2019/imgs/train'
    csv_path = 'Python/Lightning_Resnet19/DataModules/datasets/hbku2019/labels/labels_train.csv'
    dm = dataset_selector.Hbku2019DataModule(img_path, csv_path)
    # Model
    model_name = 'fc'
    num_classes = dm.classes
    model = resnet19.select_model(model_name, num_classes)
    # model_1 = resnet19.select_model('fc', num_classes, device)
    # model_2 = resnet19.select_model('conv', num_classes, device)

    # Lightning Module
    lit_model = lightning_model.PlModuleCreator(model, config, num_classes)

    # Logger
    wandb_logger = WandbLogger(
        project="lit_resnet19", name=model_name, log_model="all")
    # Trainer
    '''callbacks=[
        ModelCheckpoint(
            dirpath="Python/Lightning_Resnet19/checkpoints",
            every_n_epochs=2,
            filename='fc-{epoch}-loss:{val/Loss:.2f}-acc:{val/MulticlassAccuracy:.2f}',
            save_top_k=1,
            monitor="val/MulticlassAccuracy",
            mode="max",
        ),
    ]'''
    trainer = pl.Trainer(
        max_epochs=config['max_epochs'], logger=wandb_logger,)  # callbacks=callbacks)
    trainer.fit(lit_model, datamodule=dm)
    trainer.test(lit_model, datamodule=dm)
    wandb.finish()


if __name__ == '__main__':
    main()
