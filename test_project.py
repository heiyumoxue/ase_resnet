import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision
import torchvision.models as models
import os
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from net import *
from PIL import Image


import torch.nn.functional as F
import torch.nn as nn
from util import get_transform




import unittest
from PIL import Image
import pytorch
from torch.utils.data import DataLoader

from util import get_transform


IMG_PATH = "dataset/test/Albedo/106.png"


class TestProject(unittest.TestCase):
    def test_transform(self):
        img = Image.open(IMG_PATH)
        transform = get_transform()
        transformed_img = transform(img)
        self.assertEqual(transformed_img.shape[0], 3)
        self.assertEqual(transformed_img.shape[1], 224)
        self.assertEqual(transformed_img.shape[2], 224)
        
        
    def test_acc(self):
        img = Image.open(IMG_PATH)
        test_dataset = torchvision.datasets.ImageFolder (root=test_path, transform=get_transform())
        test_loader = DataLoader (dataset=test_dataset, batch_size=32, shuffle=True)
        self.assertEqual(transformed_img.shape[0], 3)
        self.assertEqual(transformed_img.shape[1], 224)
        self.assertEqual(transformed_img.shape[2], 224)
            
        
        