##################Build the neural network
# class ResBlock(nn.Module):
#     def __init__(self, inchannel, outchannel, stride=1):
#         super(ResBlock, self).__init__()
#         self.left = nn.Sequential(
#             nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
#             nn.BatchNorm2d(outchannel),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(outchannel)
#         )
#         self.shortcut = nn.Sequential()
#         if stride != 1 or inchannel != outchannel:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(outchannel)
#             )  
#     def forward(self, x):
#         out = self.left(x)
#         out = out + self.shortcut(x)
#         out = F.relu(out)
        
#         return out
    
# class ResNet(nn.Module):
#     def __init__(self,num_classes=2):
#         super(ResNet,self).__init__()
#         self.conv1=nn.Sequential(
#             nn.Conv2d(3,64,kernel_size=7,stride=1,padding=1,bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.Dropout(0.3)

#         )
#         self.inchannel=64
#         self.layer1=self.make_layer(ResBlock,64,2,stride=1)
#         self.layer2=self.make_layer(ResBlock,128,2,stride=2)
#         self.layer3=self.make_layer(ResBlock,256,2,stride=2)
#         self.layer4=self.make_layer(ResBlock,512,2,stride=2)
#         self.fc=nn.Linear(32768,num_classes)
    
#     def make_layer(self,block,channels,num_blocks,stride):
#         strides=[stride]+[1]*(num_blocks-1)
#         layers=[]
#         for stride in strides:
#             layers.append(block(self.inchannel,channels,stride))
#             self.inchannel=channels
        
#         return nn.Sequential(*layers)
#     def forward(self,x):
#         out=self.conv1(x)
#         out=self.layer1(out)
#         out=self.layer2(out)    
#         out=self.layer3(out)
#         out=self.layer4(out)
#         out=F.avg_pool2d(out,4)
#         out=out.view(out.size(0),-1)
#         out=self.fc(out)        
#         return out
                


#Based on this initial framework， I create：




import torch.nn as nn
import torch

from torch import Tensor
from typing import Type

class BasicBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 1,
        downsample: nn.Module = None
    ) -> None:
        super(BasicBlock, self).__init__()
        # Multiplicative factor for the subsequent conv2d layer's output channels.
        # It is 1 for ResNet18 and ResNet34.
        self.expansion = expansion
        self.downsample = downsample
        self.conv1 = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            stride=stride, 
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, 
            out_channels*self.expansion, 
            kernel_size=3, 
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels*self.expansion)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return  out

class ResNet(nn.Module):
    def __init__(
        self, 
        img_channels: int,
        num_layers: int,
        block: Type[BasicBlock],
        num_classes: int  = 1000
    ) -> None:
        super(ResNet, self).__init__()
        if num_layers == 18:
            # The following `layers` list defines the number of `BasicBlock` 
            # to use to build the network and how many basic blocks to stack
            # together.
            layers = [2, 2, 2, 2]
            self.expansion = 1
        
        self.in_channels = 64
        # All ResNets (18 to 152) contain a Conv2d => BN => ReLU for the first
        # three layers. Here, kernel size is 7.
        self.conv1 = nn.Conv2d(
            in_channels=img_channels,
            out_channels=self.in_channels,
            kernel_size=7, 
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*self.expansion, num_classes)

    def _make_layer(
        self, 
        block: Type[BasicBlock],
        out_channels: int,
        blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, 
                    out_channels*self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False 
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
        layers = []
        layers.append(
            block(
                self.in_channels, out_channels, stride, self.expansion, downsample
            )
        )
        self.in_channels = out_channels * self.expansion

        for i in range(1, blocks):
            layers.append(block(
                self.in_channels,
                out_channels,
                expansion=self.expansion
            ))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:

        # In the forward function, define how your model is going to be run, from input to output.

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)      
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        x = self.layer4(out) 
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

if __name__ == '__main__':
    tensor = torch.rand([1, 3, 224, 224])
    model = ResNet(img_channels=3, num_layers=18, block=BasicBlock, num_classes=3)
    # print(model)
    
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    # print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"{total_trainable_params:,} training parameters.")

    output = model(tensor)
    print(f"{output.shape:}Output feature size.")
    
