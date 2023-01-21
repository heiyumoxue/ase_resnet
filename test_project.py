import torch
import torch.nn.functional as F
import torch.nn as nn

import torchvision

import unittest
from PIL import Image
from torch.utils.data import DataLoader

from util import get_transform, get_acc
from net import get_resnet18


IMG_PATH = "dataset/test/Albedo/106.png"
def get_device():
    return 'cuda' if torch.cuda.is_available () else 'cpu'
device = get_device ()


class TestProject(unittest.TestCase):
    def test_transform(self):
        img = Image.open(IMG_PATH)
        transform = get_transform()
        transformed_img = transform(img)
        self.assertEqual(transformed_img.shape[0], 3)
        self.assertEqual(transformed_img.shape[1], 224)
        self.assertEqual(transformed_img.shape[2], 224)
        
    def test_acc(self):
        test_path = r'dataset/test'
        model_path = r'model_save/resnet18.pth'

        # Loading data
        test_dataset = torchvision.datasets.ImageFolder (root=test_path, transform=get_transform())
        test_loader = DataLoader (dataset=test_dataset, batch_size=32, shuffle=True)
        loss_fn = nn.CrossEntropyLoss ()

        # load Model
        model = get_resnet18().to(device)
        model.load_state_dict (torch.load(model_path))

        # get accuracy
        acc = get_acc(model, test_loader, loss_fn)
        
        self.assertGreater(acc, 0.1)

            
        
        
            
        
        