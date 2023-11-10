import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepPhys(nn.Module):
    def __init__(self):
        super(DeepPhys, self).__init__()
        
        # Convolutional layers for motion model
        self.conv_motion = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 假设输入是3个颜色通道
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Flatten()
        )

        # Fully connected layers for motion model
        self.fc_motion = nn.Sequential(
            nn.Linear(128 * 9 * 9, 128),  # 假设输入图片尺寸减半两次后为9x9
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

    def forward(self, x_motion, x_appearance):
        # Motion model
        motion_features = self.conv_motion(x_motion)
        motion_output = self.fc_motion(motion_features)

        # Appearance model
        appearance_features = self.conv_appearance(x_appearance)
        attention_weights = F.sigmoid(self.attention(appearance_features))
        attention_weights = F.normalize(attention_weights, p=1, dim=2)  # L1 normalization

        # Apply attention to motion features
        attention_applied = attention_weights * motion_features
        final_output = self.fc_motion(attention_applied.view(attention_applied.size(0), -1))

        return final_output, motion_output


