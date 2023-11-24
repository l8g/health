
from datetime import datetime
import torch
import os
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset_loader2 import create_dataloaders, create_splits, get_file_paths
from models import get_model

from rppg.config import get_config
from rppg.datasets.SegmentDataset import SegmentDataset
from utils.path_name_utils import get_dataset_directory

def train_model(model, train_loader, val_loader, criterion, optimizer, lr_sch, num_epochs, device, model_save_path='./models'):
    os.makedirs(model_save_path, exist_ok=True)
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        running_loss = 0.0

        with tqdm(train_loader, desc=f'Train Epoch {epoch+1}/{num_epochs}', total=len(train_loader)) as tepoch:
            # 训练过程
            for inputs, targets in tepoch:
                inputs = inputs.transpose(0, 1)
                d, batch_size, time_length, c, h, w = inputs.shape
                inputs = inputs.reshape(d, -1, c, h, w)
                targets = targets.reshape(-1, 1)

                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()  # 清空之前的梯度
                outputs = model(inputs[1], inputs[0])  # 前向传播
                loss = criterion(outputs, targets)
                loss.backward()  # 反向传播
                optimizer.step()  # 更新参数
                if lr_sch is not None:
                    lr_sch.step()

                running_loss += loss.item() * inputs.size(1)
                tepoch.set_description(f"Train Epoch {epoch+1}/{num_epochs}")
                tepoch.set_postfix({'': 'average loss : %0.4f | ' % loss.item()})

        epoch_loss = running_loss / len(train_loader.dataset)

        # 验证过程
        model.eval()  # 设置模型为评估模式
        val_loss = 0.0
        with torch.no_grad():  # 不计算梯度，减少内存消耗
            with tqdm(val_loader, desc=f'Val Epoch {epoch+1}/{num_epochs}', total=len(val_loader)) as tepoch:
                for inputs, targets in tepoch:
                    inputs = inputs.transpose(0, 1)
                    d, batch_size, time_length, c, h, w = inputs.shape
                    inputs = inputs.reshape(d, -1, c, h, w)
                    targets = targets.reshape(-1, 1)
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs[1], inputs[0])
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(1)
                    tepoch.set_description(f"Val Epoch {epoch+1}/{num_epochs}")
                    tepoch.set_postfix({'': 'average loss : %0.4f | ' % loss.item()})   

        val_loss /= len(val_loader.dataset)

        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}')

        current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        
        # 保存检查点
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
        }, os.path.join(model_save_path, f'checkpoint_{current_time}_epoch_{epoch+1}.pth'))

        # 如果验证损失降低，则保存最佳模型
        if val_loss < best_val_loss:
            print(f'Validation loss decreased ({best_val_loss:.4f} --> {val_loss:.4f}). Saving model ...')
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(model_save_path, 'best_model.pth'))

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

lr_sch = torch.optim.lr_scheduler.OneCycleLR(
    optimizer=optimizer,
    max_lr=cfg.fit.train.learning_rate,
    steps_per_epoch=len(train_loader),
    epochs=cfg.fit.train.epochs
)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 开始训练
train_model(model, train_loader, val_loader, criterion, optimizer, lr_sch, num_epochs, device)
