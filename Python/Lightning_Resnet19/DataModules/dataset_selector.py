'''Module with datasets class.

Implemented Datasets:

-- CIFAR-10'''

from torchvision.datasets import CIFAR10
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import kornia as K
import pytorch_lightning as pl
from PIL import Image
import numpy as np


class PreProcess(nn.Module):
    """Class to perform pre-process using Kornia on torch tensors."""

    def __init__(self) -> None:
        super().__init__()


    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: Image) -> torch.Tensor:
        '''Image prep for kornia augmentation and normalization'''
        x_tmp: np.ndarray = np.array(x)  # HxWxC
        x_out: torch.Tensor = K.image_to_tensor(x_tmp, keepdim=True)  # CxHxW
        x_out: K.enhance.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))(x_out)
        return x_out.float()


class CIFAR10DataModule(pl.LightningDataModule):
    """Class to manage CIFAR-10 dataset"""
    def __init__(self, batch_size: int = 32) -> None:
        super().__init__()
        self.path = "Python/Lightning_Resnet19/DataModules/datasets/CIFAR10"
        self.classes = 10
        self.batch_size = batch_size
        self.transform = PreProcess()
        #self.train_dataset, self.val_dataset,self.test_dataset = {}, {}, {}


    def prepare_data(self) -> None:
        CIFAR10(root=self.path, train=True, download=True)
        CIFAR10(root=self.path, train=False, download=True)


    def setup(self, stage: str = None) -> None:
        if stage == 'fit' or stage is None:
            cifar10_full = CIFAR10(
                root=self.path, train=True, transform=self.transform)
            self.train_dataset, self.val_dataset = random_split(
                cifar10_full, [45000, 5000])


        if stage == 'test' or stage is None:
            self.test_dataset = CIFAR10(
                root=self.path, train=False, transform=self.transform)


    def train_dataloader(self):
        '''Return train dataloader of CIFAR10'''
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                           shuffle=True, num_workers=7, persistent_workers=True)


    def val_dataloader(self):
        '''Return validation dataloader of CIFAR10'''
        return DataLoader(self.val_dataset, batch_size=self.batch_size*4,
                          num_workers=7, persistent_workers=True)


    def test_dataloader(self):
        '''Return test dataloader of CIFAR10'''
        return DataLoader(self.test_dataset, batch_size=self.batch_size*4, num_workers=7)
