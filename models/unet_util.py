""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Applies two consecutive sets of operations: 
    (Convolution => Batch Normalization => ReLU activation) * 2.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels after the second convolution.
        mid_channels (int, optional): Number of channels used in the intermediate 
                                      convolution. If not provided, defaults to out_channels.
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            # padding=1 保证输入输出的宽度和高度一致
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
        """
        Forward pass through the DoubleConv module.

        Args:
            x (Tensor): Input tensor of shape (N, C_in, H, W), where:
                - N is the batch size,
                - C_in is the number of input channels,
                - H is the height of the input feature map,
                - W is the width of the input feature map.
        Returns:
            Tensor: Output tensor of shape (N, C_out, H, W), where:
                - N is the batch size,
                - C_out is the number of output channels (as specified by out_channels),
                - H and W are the height and width of the output feature map (same as input).
        """
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), # 如果输入的特征图尺寸是 H*W ，经过最大池化后，特征图的尺寸将变为  H/2 * W/2 ，而通道数不变
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)  # 增加特征图的空间分辨率（高度和宽度）
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        Forward pass for the Up module in U-Net.

        This method takes two input feature maps: 
        - `x1`: The lower resolution feature map, which is upsampled to match the size of `x2`.
        - `x2`: The higher resolution feature map from the encoder (skip connection).

        Args:
            x1 (Tensor): Low-resolution input tensor of shape (N, C_in1, H1, W1) where:
                        - N is the batch size,
                        - C_in1 is the number of input channels (after being halved during upsampling),
                        - H1 and W1 are the height and width after upsampling.
            x2 (Tensor): High-resolution input tensor of shape (N, C_in2, H2, W2) where:
                        - C_in2 is the number of input channels from the encoder,
                        - H2 and W2 are the spatial dimensions (which are larger than H1 and W1).

        Returns:
            Tensor: Output tensor of shape (N, C_out, H2, W2) where:
                    - N is the batch size,
                    - C_out is the number of output channels after the convolutional layer,
                    - H2 and W2 are the height and width, which match those of `x2`.
        """
        x1 = self.up(x1) # Upsample x1, channel count reduced by half
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        # Pad x1 to match the size of x2
        """
        if you have padding issues, see
        https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        """
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # Concatenate along the channel dimension
        x = torch.cat([x2, x1], dim=1)  # Concatenate feature maps, channel count increases
        
        # Apply convolution to reduce the channel count and process features
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) # 1x1 卷积核的作用在于它只对通道进行混合或转换，而不会影响特征图的空间分辨率。

    def forward(self, x):
        return self.conv(x)