import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl

class SegmentationLit(pl.LightningModule):
    def __init__(self, hparams):
        """

        Args:
            hparams:

        Example:
            >>>
        """
        super(SegmentationLit, self).__init__()
        self.hparams = hparams

    def training_step(self, *args, **kwargs):
        pass

    
