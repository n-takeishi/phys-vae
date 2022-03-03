import torch
import torch.nn as nn

from . import utils
# import utils

# most part taken from https://github.com/usuyama/pytorch-unet


class UNet(nn.Module):
    def __init__(self, k:int=2):
        super().__init__()

        # H + 2p -k +1
        self.dconv_down1 = nn.Sequential(
            nn.Conv2d(4, k*4, 3, padding=0), nn.BatchNorm2d(k*4), nn.ReLU(inplace=True),
            nn.Conv2d(k*4, k*4, 4, padding=0), nn.BatchNorm2d(k*4), nn.ReLU(inplace=True)
        )

        self.dconv_down2 = nn.Sequential(
            nn.Conv2d(k*4, k*8, 3, padding=1), nn.BatchNorm2d(k*8), nn.ReLU(inplace=True),
            nn.Conv2d(k*8, k*8, 3, padding=1), nn.BatchNorm2d(k*8), nn.ReLU(inplace=True)
        )

        self.dconv_down3 = nn.Sequential(
            nn.Conv2d(k*8, k*16, 3, padding=1), nn.BatchNorm2d(k*16), nn.ReLU(inplace=True),
            nn.Conv2d(k*16, k*16, 3, padding=1), nn.BatchNorm2d(k*16), nn.ReLU(inplace=True)
        )
        self.dconv_down4 = nn.Sequential(
            nn.Conv2d(k*16, k*32, 3, padding=1), nn.BatchNorm2d(k*32), nn.ReLU(inplace=True),
            nn.Conv2d(k*32, k*32, 3, padding=1), nn.BatchNorm2d(k*32), nn.ReLU(inplace=True)
        )

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = nn.Sequential(
            nn.Conv2d(k*16+k*32, k*16, 3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(k*16),
            nn.Conv2d(k*16, k*16, 3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(k*16)
        )

        self.dconv_up2 = nn.Sequential(
            nn.Conv2d(k*8+k*16, k*8, 3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(k*8),
            nn.Conv2d(k*8, k*8, 3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(k*8)
        )

        self.dconv_up1 = nn.Sequential(
            nn.Conv2d(k*4+k*8, k*4, 3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(k*4),
            nn.Conv2d(k*4, k*4, 3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(k*4)
        )

        self.conv_last = nn.Conv2d(k*4, 3, 2, padding=3)


    def forward(self, x):
        conv1 = self.dconv_down1(x) # k*4, 64, 64
        x = self.maxpool(conv1) # k*4, 32, 32

        conv2 = self.dconv_down2(x) # k*8, 32, 32
        x = self.maxpool(conv2) # k*8, 16, 16

        conv3 = self.dconv_down3(x) # k*16, 16, 16
        x = self.maxpool(conv3) # k*16, 8, 8

        x = self.dconv_down4(x) # k*32, 8, 8

        x = self.upsample(x) # k*32, 16, 16
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x) # k*16, 16, 16

        x = self.upsample(x) # k*16, 32, 32
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x) # k*8, 32, 32

        x = self.upsample(x) # k*8, 64, 64
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x) # k*4, 64, 64

        out = self.conv_last(x)

        return out
