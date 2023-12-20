'''Module with lit model and augmentation logic'''
import torch
from torch import nn, optim
import torchmetrics
import pytorch_lightning as pl
import kornia as K


class DataAugmentation(nn.Module):
    '''Class to perform data augmentation using Kornia on torch tensors.'''
    def __init__(self) -> None:
        super().__init__()
        self._max_val: float = 255.0
        self.transforms = nn.Sequential(
            K.augmentation.RandomHorizontalFlip(p=0.5),
            K.augmentation.RandomVerticalFlip(p=0.5))

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Function to apply online augmentation to batch x'''
        x_out = self.transforms(x)  # BxCxHxW
        return x_out


class PlModuleCreator(pl.LightningModule):
    '''Class that contains lit model logic.'''
    def __init__(self, model: nn.Module, config: dict, num_classes: int) -> None:
        super().__init__()
        self.model = model
        self.config = config
        self.transform = DataAugmentation()
        #Metrics class
        metrics = torchmetrics.MetricCollection(
            torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes),
            torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes),
            torchmetrics.AveragePrecision(
            task="multiclass", num_classes=num_classes),
            torchmetrics.Precision(
            task="multiclass", num_classes=num_classes),
            torchmetrics.Recall(
            task="multiclass", num_classes=num_classes),
            )
        self.test_metrics = metrics.clone(prefix='test/')
        self.val_metrics = metrics.clone(prefix='val/')
        self.train_metrics = metrics.clone(prefix='train/')

        # Loss function
        if self.config['loss'] == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
        self.save_hyperparameters(ignore=['model'])


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


    def training_step(self, batch: torch.utils.data.DataLoader, batch_idx: int) -> torch.Tensor:
        x, y = batch
        x_aug = self.transform(x)
        y_hat = self.forward(x_aug)
        loss = self.criterion(y_hat, y)
        self.log("train/Loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.train_metrics.update(y_hat, y)
        self.log_dict(self.train_metrics, on_step=True, on_epoch=True)
        return loss


    def validation_step(self, batch: torch.utils.data.DataLoader, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log('val/Loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.val_metrics.update(y_hat, y)
        self.log_dict(self.val_metrics, on_step=True, on_epoch=True)
        return loss


    def test_step(self, batch: torch.utils.data.DataLoader, batch_idx: int) -> None:
        x, y = batch
        y_hat = self.forward(x)
        self.test_metrics.update(y_hat, y)
        self.log_dict(self.test_metrics, on_step=True, on_epoch=True)
        return None


    def configure_optimizers(self) -> optim.Optimizer:
        if self.config['optimizer'] == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.config['lr'],)
                                   #weight_decay=self.config['weight_decay'])
        elif self.config['optimizer'] == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=self.config['lr'],
                                  momentum=self.config['momentum'],
                                  weight_decay=self.config['weight_decay'])

        if self.config['scheduler'] == 'step_lr':
            scheduler = optim.lr_scheduler.StepLR(optimizer,
                                                  step_size=self.config['step_size'],
                                                  gamma=self.config['gamma'])
            return [optimizer], [scheduler]
        return optimizer
