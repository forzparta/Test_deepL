'''Module with datasets class.

Implemented Datasets:

-- CIFAR-10'''

from torchvision.datasets import CIFAR10
from torchvision import transforms
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, Dataset
import kornia as K
import pytorch_lightning as pl
from PIL import Image
import numpy as np
import pandas as pd


class PreProcessCIFAR(nn.Module):
    """Class to perform pre-process using Kornia on torch tensors."""

    def __init__(self) -> None:
        super().__init__()

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: Image) -> torch.Tensor:
        '''Image prep for kornia augmentation and normalization'''
        x_tmp: np.ndarray = np.array(x)  # HxWxC
        x_out: torch.Tensor = K.image_to_tensor(x_tmp, keepdim=True)  # CxHxW
        x_out: K.enhance.Normalize(
            (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))(x_out)
        return x_out.float()


class PreProcessHbku2019(nn.Module):
    """Class to perform pre-process using Kornia on torch tensors."""

    def __init__(self) -> None:
        super().__init__()
        # Img model input size
        self.im_size = 224
        self.transform = transforms.Compose([
            transforms.Resize((self.im_size, self.im_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: Image) -> torch.Tensor:
        '''Image prep for kornia augmentation and normalization'''
        # x_tmp = transforms.Resize((self.im_size, self.im_size))(x)
        # x_tmp: np.ndarray = np.array(x_tmp)  # HxWxC
        # x_out: torch.Tensor = K.image_to_tensor(x_tmp, keepdim=True)  # CxHxW
        # x_out: K.enhance.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(x_out)
        x_out = self.transform(x)
        return x_out


class CIFAR10DataModule(pl.LightningDataModule):
    """Class to manage CIFAR-10 dataset"""

    def __init__(self, batch_size: int = 32) -> None:
        super().__init__()
        self.path = "Python/Lightning_Resnet19/DataModules/datasets/CIFAR10"
        self.classes = 10
        self.batch_size = batch_size
        self.transform = PreProcessCIFAR()
        # self.train_dataset, self.val_dataset,self.test_dataset = {}, {}, {}

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
                          shuffle=True, num_workers=4, persistent_workers=True)

    def val_dataloader(self):
        '''Return validation dataloader of CIFAR10'''
        return DataLoader(self.val_dataset, batch_size=self.batch_size*4,
                          num_workers=4, persistent_workers=True)

    def test_dataloader(self):
        '''Return test dataloader of CIFAR10'''
        return DataLoader(self.test_dataset, batch_size=self.batch_size*4,
                          num_workers=4)


class Hbku2019DataModule(pl.LightningDataModule):
    """Class to manage custom local datasets"""

    def __init__(self, imgs_path: str, csv_path: str, batch_size: int = 32) -> None:
        super().__init__()
        self.img_path = imgs_path
        self.csv_path = csv_path
        self.classes = 80
        self.batch_size = batch_size
        self.transform = PreProcessHbku2019()

    def prepare_data(self) -> None:
        CustomDatasetFromCSV(self.img_path, self.csv_path,
                             self.transform, train=True)
        CustomDatasetFromCSV(self.img_path, self.csv_path,
                             self.transform, train=False)

    def setup(self, stage: str = None) -> None:
        if stage == 'fit' or stage is None:
            custom_full = CustomDatasetFromCSV(
                self.img_path, self.csv_path, self.transform, train=True)
            self.train_dataset, self.val_dataset = random_split(
                custom_full, [0.7, 0.3])

        if stage == 'test' or stage is None:
            self.test_dataset = CustomDatasetFromCSV(
                self.img_path, self.csv_path, self.transform, train=False)

    def train_dataloader(self):
        '''Return train dataloader'''
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=4, persistent_workers=True, pin_memory=True)

    def val_dataloader(self):
        '''Return validation dataloader'''
        return DataLoader(self.val_dataset, batch_size=self.batch_size*8,
                          num_workers=4, persistent_workers=True, pin_memory=True)

    def test_dataloader(self):
        '''Return test dataloader'''
        return DataLoader(self.test_dataset, batch_size=self.batch_size*8, num_workers=4,
                          persistent_workers=True)


class CustomDatasetFromCSV(Dataset):
    '''Custom dataset from local using csv and folder with images'''

    def __init__(self, imgs_path, csv_path, transformations, train):
        """
        Args:
            csv_path (string): path to csv file
            transformations: pytorch transforms for transforms and tensor conversion
            train: flag to determine if train or val set
        """
        self.path = imgs_path
        # Transforms
        self.transforms = transformations
        # Read the csv file
        self.data_info = pd.read_csv(csv_path)
        # self.data_info = np.random.shuffle(self.data_info)

        if train:
            self.image_arr = (self.data_info.iloc[:90000, 0])
        else:
            self.image_arr = (self.data_info.iloc[90000:, 0])
        self.image_arr = np.asarray(self.image_arr)
        # Second column is the labels
        if train:
            self.label_arr = np.asarray(self.data_info.iloc[:90000, 1:])
        else:
            self.label_arr = np.asarray(self.data_info.iloc[90000:, 1:])

        # Calculate len
        self.data_len = len(self.label_arr)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]

        # Open image
        img_as_img = Image.open(
            self.path + '/' + single_image_name).convert('RGB')
        if self.transforms is not None:
            img_as_tensor = self.transforms(img_as_img)

        single_image_label = self.label_arr[index]
        return (img_as_tensor, single_image_label)

    def __len__(self):
        return self.data_len


class Hbku2019Debug(Dataset):
    '''Custom dataset from local using csv and folder with images'''

    def __init__(self, imgs_path, csv_path, transformations, train):
        """
        Args:
            csv_path (string): path to csv file
            transformations: pytorch transforms for transforms and tensor conversion
            train: flag to determine if train or val set
        """
        self.path = imgs_path
        # Transforms
        self.transforms = transformations
        # Read the csv file
        self.data_info = pd.read_csv(csv_path)
        # self.data_info = np.random.shuffle(self.data_info)

        if train:
            self.image_arr = (self.data_info.iloc[:, 0])
        else:
            self.image_arr = (self.data_info.iloc[90000:, 0])
        self.image_arr = np.asarray(self.image_arr)
        # Second column is the labels
        if train:
            self.label_arr = np.asarray(self.data_info.iloc[:, 1:])
        else:
            self.label_arr = np.asarray(self.data_info.iloc[90000:, 1:])

        # Calculate len
        self.data_len = len(self.label_arr)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]

        # Open image
        img_as_img = Image.open(
            self.path + '/' + single_image_name).convert('RGB')
        if self.transforms is not None:
            img_as_tensor = self.transforms(img_as_img)

        single_image_label = self.label_arr[index]
        return (img_as_tensor, single_image_label)

    def __len__(self):
        return self.data_len
