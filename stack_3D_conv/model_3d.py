import torch
import torch.nn as nn 
import torch.nn.functional as F
from torchsummary import summary

class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(1, 1, 1)):
        super(ResidualBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), 
                               stride=stride, padding=(1, 1, 1), bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), 
                               stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

        # shortcut connection for matching dimensions
        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != (1, 1, 1):
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += shortcut
        return F.relu(x)
    
class Gesture3DNet(nn.Module):
    def __init__(self):
        super(Gesture3DNet, self).__init__()

        self.conv1 = nn.Conv3d(20, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d((1, 2, 2))

        self.res_block1 = ResidualBlock3D(32, 64, stride=(1, 2, 2))
        self.res_block2 = ResidualBlock3D(64, 128, stride=(1, 2, 2))
        self.res_block3 = ResidualBlock3D(128, 256, stride=(1, 2, 2))

        self.dropout_conv = nn.Dropout3d(0.3)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc1 = nn.Linear(256, 128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 7)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))

        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.dropout_conv(x)
        x = self.global_avg_pool(x)

        x = torch.flatten(x, start_dim=1)
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Gesture3DNet().to(device)
    summary(model, (20, 3, 240, 320), device=device)

    