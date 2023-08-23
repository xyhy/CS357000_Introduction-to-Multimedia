import os
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

import torch
from torch import Tensor
from torch.utils.data import Dataset
from pytorch_lightning import LightningDataModule
from torchvision.datasets.folder import default_loader
import torchvision
from torch.utils.data import DataLoader
import glob

class FingerPrinterDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_names = glob.glob(self.img_dir+"/*.BMP")

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = default_loader(img_path)
        if self.transform:
            image = self.transform(image)
        return image