import os
from torch.utils.data import DataLoader, random_split

# 获取指定目录下所有的.hdf5文件
def get_file_paths(directory):
    return [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.hdf5')]

# 创建数据集分割的函数
def create_splits(file_paths, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    total_size = len(file_paths)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    # 确保总数正确
    assert train_size + val_size + test_size == total_size
    
    return random_split(file_paths, [train_size, val_size, test_size])

# 创建DataLoaders的函数
def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
