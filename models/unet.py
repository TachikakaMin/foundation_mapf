""" Full assembly of the parts to form the complete network """

from .unet_util import *

class UNet(nn.Module):
    """最小能处理的输入尺寸为 16*16
    """
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.input_conv = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down1_1 = (Down(128, 128))
        # self.down1_2 = (Down(128, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.output_conv = (OutConv(64, n_classes))
        # 将模型的输出转换为概率分布，dim=-1 表示在最后一个维度上进行转换
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.input_conv(x)
        x2 = self.down1(x1)
        # x2 = self.down1_1(x2)
        # x2 = self.down1_2(x2)
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
        """checkpoint通过在前向传播过程中
           保存某些关键的激活值，而不是保存所有中间层的激活值。
           这样，在反向传播时，需要重新计算那些未保存的激活值。
           尽管增加了计算开销（因为有些前向计算需要重新执行），但节省了大量的显存
        """
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