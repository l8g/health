


import torch
from dataset_loader2 import create_dataloaders, create_splits, get_file_paths
from models import get_model
from rppg.config import get_config
from torch import nn
from rppg.datasets.SegmentDataset import SegmentDataset

import matplotlib.pyplot as plt
import numpy as np


from utils.path_name_utils import get_dataset_directory

cfg = get_config('./configs/base_config.yaml')

# 模型实例化
model = get_model(fit_cfg=cfg.fit)

# 损失函数实例化
criterion = nn.MSELoss() # 或者是任何其他的损失函数

# 优化器实例化
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.fit.train.learning_rate)

# 获取文件路径
directory = get_dataset_directory(cfg)
file_paths = get_file_paths(directory)
# file_paths = file_paths[:10]

# 分割数据集
train_files, val_files, test_files = create_splits(file_paths)

# 根据文件分割创建各自的SegmentDataset实例
train_dataset = SegmentDataset(train_files)
val_dataset = SegmentDataset(val_files)
test_dataset = SegmentDataset(test_files)

batch_size = cfg.fit.train.batch_size
num_epochs = cfg.fit.train.epochs
# 创建DataLoaders
train_loader, val_loader, test_loader = create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size)


for inputs, targets in train_loader:
    print(inputs.shape)
    print(targets.shape)

    inputs = inputs.transpose(0, 1)
    d, batch_size, time_length, c, h, w = inputs.shape
    inputs = inputs.reshape(d, -1, c, h, w)
    targets = targets.reshape(-1, 1)
    print(inputs.shape)
    print(targets.shape)
    
    image = inputs[0][100].permute(1, 2, 0)
    plt.imshow(image)
    plt.show()

    
    image = inputs[1][100].permute(1, 2, 0)
    print(image.mean(), image.std(), image.min(), image.max())
    image = (image - image.min()) / (image.max() - image.min())
    plt.imshow(image)
    plt.show()

    break

