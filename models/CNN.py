import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(CNN, self).__init__()
        
        # Initial conv block
        self.conv1 = nn.Conv2d(n_channels, 64, kernel_size=3, padding=1)  # Changed to 96 channels
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual blocks
        self.res_block1 = ResidualBlock(64, 128)  # Adjusted channels
        self.res_block2 = ResidualBlock(128, 256)
        self.res_block3 = ResidualBlock(256, 512)
        self.res_block4 = ResidualBlock(512, 1024)
        self.res_block5 = ResidualBlock(1024, 512)
        self.res_block6 = ResidualBlock(512, 256)
        self.res_block7 = ResidualBlock(256, 128)
        self.res_block8 = ResidualBlock(128, 64)
        # Final conv to match n_classes
        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Initial conv
        x = self.relu(self.bn1(self.conv1(x)))
        
        # Residual blocks
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.res_block5(x)
        x = self.res_block6(x)
        x = self.res_block7(x)
        x = self.res_block8(x)
        # Final conv and softmax
        logits = self.final_conv(x)
        prob = self.softmax(logits)
        return logits, prob

    def save_model(self, path):
        torch.save(self.state_dict(), path)
    
    def load_model(self, path):
        self.load_state_dict(torch.load(path))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv_block(x)
        out += self.shortcut(residual)
        out = self.relu(out)
        return out

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    
    n_channels = 6
    n_classes = 5

    model = CNN(n_channels, n_classes)
    print(f"Number of parameters in the model: {count_parameters(model)}")