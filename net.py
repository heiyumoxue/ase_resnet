import torchvision.models as models
from torch import nn
from Resnet18_pytorch import *

# 这里使用迁移学习便于训练，不需要的话把pretrain设置为False
def get_resnet():
    model = models.resnet50 (pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear (num_ftrs, 5)
    return model


def get_resnext():
    model = models.resnext50_32x4d (pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear (num_ftrs, 5)
    return model

def get_resnet18():
    torch.manual_seed(42)
    model = ResNet(img_channels=3, num_layers=18, block=BasicBlock, num_classes=3)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear (num_ftrs, 5)
    return model



if __name__ == '__main__':
    print (get_resnet ())




