'''Module with lit model and augmentation logic'''
import torch
from torch import nn, optim
import torchmetrics
import pytorch_lightning as pl
import kornia as K
from torchmetrics.classification import MultilabelAveragePrecision
from torchmetrics.classification import MultilabelF1Score
from torchmetrics.classification import MultilabelAccuracy
from torchmetrics.classification import MultilabelAUROC


class DataAugmentation(nn.Module):
    '''Class to perform data augmentation using Kornia on torch tensors.'''

    def __init__(self) -> None:
        super().__init__()
        self.transforms = nn.Sequential(
            K.augmentation.RandomHorizontalFlip(p=0.3),
            K.augmentation.RandomVerticalFlip(p=0.3))

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
        # Metrics class
        '''metrics = torchmetrics.MetricCollection(
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
            )'''

        # self.test_metrics = metrics.clone(prefix='test/')
        # self.val_metrics = metrics.clone(prefix='val/')
        # self.train_metrics = metrics.clone(prefix='train/')
        # Loss function
        if self.config['loss'] == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
            self.test_mAP = torchmetrics.AveragePrecision(
                task="multiclass", num_classes=num_classes)
            self.val_mAP = torchmetrics.AveragePrecision(
                task="multiclass", num_classes=num_classes)
            self.train_mAP = torchmetrics.AveragePrecision(
                task="multiclass", num_classes=num_classes)
        elif self.config['loss'] == 'bin_cross_entropy':
            self.criterion = nn.BCEWithLogitsLoss()
            self.test_mAP = MultilabelAveragePrecision(
                num_labels=num_classes, average='macro')
            self.val_mAP = MultilabelAveragePrecision(
                num_labels=num_classes, average='macro')
            self.train_mAP = MultilabelAveragePrecision(
                num_labels=num_classes, average='macro')

            self.test_acc = MultilabelAccuracy(
                num_labels=num_classes, average='macro')
            self.val_acc = MultilabelAccuracy(
                num_labels=num_classes, average='macro')
            self.train_acc = MultilabelAccuracy(
                num_labels=num_classes, average='macro')

            self.test_f1 = MultilabelF1Score(
                num_labels=num_classes, average='macro')
            self.val_f1 = MultilabelF1Score(
                num_labels=num_classes, average='macro')
            self.train_f1 = MultilabelF1Score(
                num_labels=num_classes, average='macro')

            self.test_auroc = MultilabelAUROC(
                num_labels=num_classes, average='macro')
            self.val_auroc = MultilabelAUROC(
                num_labels=num_classes, average='macro')
            self.train_auroc = MultilabelAUROC(
                num_labels=num_classes, average='macro')
        self.save_hyperparameters(ignore=['model'])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: torch.utils.data.DataLoader, batch_idx: int) -> torch.Tensor:
        x, y= batch
        x_aug = self.transform(x)
        y_hat = self.forward(x_aug)
        if self.config['loss'] == 'bin_cross_entropy':
            loss = self.criterion(y_hat, y)
            self.train_mAP(y_hat, y.to(torch.int64))
            self.train_acc(y_hat, y.to(torch.int64))
            self.train_auroc(y_hat, y.to(torch.int64))
            self.train_f1(y_hat, y.to(torch.int64))

            self.log("train/mAP", self.train_mAP,
                 prog_bar=True, on_step=True, on_epoch=True)
            self.log("train/acc", self.train_acc,
                 prog_bar=True, on_step=True, on_epoch=True)
            self.log("train/auroc", self.train_auroc,
                 prog_bar=True, on_step=True, on_epoch=True)
            self.log("train/f1", self.train_f1,
                 prog_bar=True, on_step=True, on_epoch=True)

        self.log("train/Loss", loss, prog_bar=True,
                 on_step=True, on_epoch=False)
        # self.log_dict(self.train_metrics, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: torch.utils.data.DataLoader, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)

        if self.config['loss'] == 'bin_cross_entropy':
            loss = self.criterion(y_hat, y)
            self.val_mAP(y_hat, y.to(torch.int64))
            self.val_acc(y_hat, y.to(torch.int64))
            self.val_auroc(y_hat, y.to(torch.int64))
            self.val_f1(y_hat, y.to(torch.int64))

            self.log("val/mAP", self.val_mAP,
                on_step=False, on_epoch=True)
            self.log("val/acc", self.val_acc,
                on_step=False, on_epoch=True)
            self.log("val/auroc", self.val_auroc,
                on_step=False, on_epoch=True)
            self.log("val/f1", self.val_f1,
                on_step=False, on_epoch=True)

        self.log('val/Loss', loss, prog_bar=True, on_step=False,
                  on_epoch=True)
        # self.val_metrics.update(y_hat, y)
        # self.log_dict(self.val_metrics, on_step=True, on_epoch=True)
        return loss

    def test_step(self, batch: torch.utils.data.DataLoader, batch_idx: int) -> None:
        x, y = batch
        y_hat = self.forward(x)
        if self.config['loss'] == 'bin_cross_entropy':
            self.test_mAP(y_hat, y.to(torch.int64))
            self.test_acc(y_hat, y.to(torch.int64))
            self.test_auroc(y_hat, y.to(torch.int64))
            self.test_f1(y_hat, y.to(torch.int64))

            self.log("test/mAP", self.test_mAP,
                on_step=False, on_epoch=True)
            self.log("test/acc", self.test_acc,
                on_step=False, on_epoch=True)
            self.log("test/auroc", self.test_auroc,
                on_step=False, on_epoch=True)
            self.log("test/f1", self.test_f1,
                on_step=False, on_epoch=True)
        # self.log_dict(self.test_metrics, on_step=True, on_epoch=True)
        return None

    def configure_optimizers(self) -> optim.Optimizer:
        if self.config['optimizer'] == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(), lr=self.config['lr'],)
            # weight_decay=self.config['weight_decay'])
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
