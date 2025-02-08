import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class SpatialStream(nn.Module):
    def __init__(self, num_classes):
        super(SpatialStream, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2,2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2,2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2,2)

        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(2,2)


        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 256)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.pool5(F.relu(self.bn5(self.conv5(x))))
        x = self.global_avg_pool(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

class TemporalStream(nn.Module):
    def __init__(self, num_classes):
        super(TemporalStream, self).__init__()
        
        self.conv1 = nn.Conv3d(2, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1)
        self.bn1 = nn.BatchNorm3d(32)

        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1)
        self.bn2 = nn.BatchNorm3d(64)

        self.conv3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.pool3 = nn.MaxPool3d((2, 2, 2), (2, 2, 2))

        self.conv4 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1)
        self.bn4 = nn.BatchNorm3d(256)

        self.conv5 = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1)
        self.bn5 = nn.BatchNorm3d(512)
        self.pool5 = nn.MaxPool3d((2, 2, 2), (2, 2, 2))

        self.global_avg_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.fc1 = nn.Linear(512*7, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool5(x)
        
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class TwoStreamModel(nn.Module):
    def __init__(self, num_classes):
        super(TwoStreamModel, self).__init__()
        self.spatial_stream = SpatialStream(num_classes)        
        self.temporal_stream = TemporalStream(num_classes)

    def forward(self, rgb_frame, optical_flow):
        spatial_out = self.spatial_stream(rgb_frame) 
        temporal_out = self.temporal_stream(optical_flow)
    
        #late fusion
        output = 0.3 * spatial_out + 0.7 * temporal_out
        # print(f"SPATIAL: {spatial_out}")
        # print(F"TEMPORAL : {temporal_out}")
        return output

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TwoStreamModel(num_classes=7).to(device)

    # Separate summaries for spatial and temporal streams
    print("Spatial Stream Summary:")
    spatial_model = model.spatial_stream
    summary(spatial_model, (3, 224, 224), device=device)

    print("\nTemporal Stream Summary:")
    temporal_model = model.temporal_stream
    summary(temporal_model, (2, 30, 224, 224), device=device)