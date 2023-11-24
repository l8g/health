import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepPhys(nn.Module):
    def __init__(self):
        super(DeepPhys, self).__init__()
        
        # Convolutional layers for motion model
        self.conv_motion = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # Output size: 128x128
            nn.Tanh(),
            nn.AvgPool2d(2),  # Output size: 64x64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.AvgPool2d(2),  # Output size: 32x32
            nn.Flatten()
        )

        # Adjust the input size for the fully connected layer
        self.fc_motion = nn.Sequential(
            nn.Linear(64 * 32 * 32, 128),  # Adjusted for 32x32 feature maps
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # Convolutional layers for appearance model
        self.conv_appearance = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.Tanh(),
        )

        # Attention mechanism
        self.attention = nn.Conv2d(128, 1, kernel_size=1)

    def forward(self, x_motion, x_appearance): # torch.Size([180, 3, 128, 128])
        motion_features = self.conv_motion(x_motion)  # 形状为 [batch_size, channels, height, width]
        motion_output = self.fc_motion(motion_features)

        return motion_output, motion_output

