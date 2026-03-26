""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """残差块：Conv3x3 → BN → ReLU → Conv3x3 → BN + shortcut → ReLU
    Residual block with two 3x3 convolutions and identity shortcut.
    Input and output channel counts must be equal."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + x)


class ResStage(nn.Module):
    """通道变换 + N 个 ResBlock。替代原来的 DoubleConv。
    Channel-changing conv followed by N ResBlocks. Replaces DoubleConv."""

    def __init__(self, in_channels, out_channels, num_blocks=1):
        super().__init__()
        # 第一层：变换通道数
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        # 后续：num_blocks 个残差块，通道数不变
        self.blocks = nn.Sequential(
            *[ResBlock(out_channels) for _ in range(num_blocks)]
        )

    def forward(self, x):
        return self.blocks(self.proj(x))


class DoubleConv(nn.Module):
    """(Convolution => BN => ReLU) * 2. Kept for backward compatibility."""

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


class Down(nn.Module):
    """MaxPool → ResStage (or DoubleConv when num_blocks=0)"""

    def __init__(self, in_channels, out_channels, num_blocks=1):
        super().__init__()
        conv = ResStage(in_channels, out_channels, num_blocks) if num_blocks > 0 else DoubleConv(in_channels, out_channels)
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), conv)

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then ResStage (or DoubleConv when num_blocks=0)"""

    def __init__(self, in_channels, out_channels, bilinear=True, num_blocks=1):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            conv_in = in_channels
            mid = in_channels // 2
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            conv_in = in_channels

        if num_blocks > 0:
            self.conv = ResStage(conv_in, out_channels, num_blocks)
        else:
            self.conv = DoubleConv(conv_in, out_channels, mid if bilinear else None)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
