""" Full assembly of the parts to form the complete network """

from .unet_util import *


class UNet(nn.Module):
    """最小能处理的输入尺寸为 16*16
    blocks_per_stage: 每个 stage 的 ResBlock 数量，0 则退化为原始 DoubleConv
    """
    def __init__(self, n_channels, n_classes, first_layer_channels=64, bilinear=False, blocks_per_stage=1):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        C = first_layer_channels
        factor = 2 if bilinear else 1
        B = blocks_per_stage

        if B > 0:
            self.input_conv = ResStage(n_channels, C, B)
        else:
            self.input_conv = DoubleConv(n_channels, C)
        self.down1 = Down(C, C * 2, B)
        self.down2 = Down(C * 2, C * 4, B)
        self.down3 = Down(C * 4, C * 8, B)
        self.down4 = Down(C * 8, C * 16 // factor, B)
        self.up1 = Up(C * 16, C * 8 // factor, bilinear, B)
        self.up2 = Up(C * 8, C * 4 // factor, bilinear, B)
        self.up3 = Up(C * 4, C * 2 // factor, bilinear, B)
        self.up4 = Up(C * 2, C, bilinear, B)
        self.output_conv = OutConv(C, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.input_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.output_conv(x)
        prob = self.softmax(logits)
        return logits, prob

    def use_checkpointing(self):
        self.input_conv = torch.utils.checkpoint(self.input_conv)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.output_conv = torch.utils.checkpoint(self.output_conv)
