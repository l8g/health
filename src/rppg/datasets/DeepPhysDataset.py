

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class DeepPhysDataset(Dataset):
    def __init__(self, appearance_data, motion_data, target):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.a = appearance_data
        self.m = motion_data
        self.label = target.reshape(-1, 1)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        
        appearance_data = torch.tensor(np.transpose(self.a[index], (2, 0, 1)), dtype=torch.float32)
        motion_data = torch.tensor(np.transpose(self.m[index], (2, 0, 1)), dtype=torch.float32)
        label = torch.tensor(self.label[index], dtype=torch.float32)

        if torch.cuda.is_available():
            appearance_data = appearance_data.cuda()
            motion_data = motion_data.cuda()
            label = label.cuda()

        return appearance_data, motion_data, label
    
    def __len__(self):
        return len(self.label)