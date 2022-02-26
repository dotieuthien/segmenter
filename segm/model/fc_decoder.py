import math
from tkinter.messagebox import NO
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2
    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """Upscaling then double conv
    """

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # If bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels // 2, out_channels)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)

        # input is CxHxW
        # diffY = x1.size()[2] - x2.size()[2]
        # diffX = x1.size()[3] - x2.size()[3]

        # x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        if x2 is not None:
            H, W = x1.size()[2], x1.size()[3]
            x2 = F.interpolate(x2, size=(H, W), mode="nearest")

            x1 = torch.cat([x1, x2], dim=1)

        return self.conv(x1)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class DecoderUNet(nn.Module):
    def __init__(self, n_channels=512, n_classes=7, bilinear=False):
        """Decoder to predict mask using unet architecture

        Args:
            n_channels (int): number of channels in encoder's output
            n_classes (int): number of classes
            bilinear (bool, optional): _description_. Defaults to False.
        """
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, n_channels)
        factor = 1 if bilinear else 1
        self.up1 = Up(n_channels, (n_channels // 2) // factor, bilinear)
        self.up2 = Up((n_channels //2), (n_channels // 4) // factor, bilinear)
        self.up3 = Up((n_channels // 4), (n_channels // 8) // factor, bilinear)
        self.up4 = Up((n_channels // 8), (n_channels // 16), bilinear)

        self.fc = nn.Sequential(
            nn.Linear((n_channels // 16), (n_channels // 16)),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Linear((n_channels // 16), n_classes),
            )

    def forward(self, x, im_size):
        x = x.contiguous().view(-1, self.n_channels, 32, 32)
        x1 = self.inc(x)
        x2 = self.up1(x1)
        x3 = self.up2(x2)
        x4 = self.up3(x3)
        x5 = self.up4(x4)

        x = x5.permute(0, 2, 3, 1)
        logits = self.fc(x)
        logits = logits.permute(0, 3, 1, 2)
        return logits

