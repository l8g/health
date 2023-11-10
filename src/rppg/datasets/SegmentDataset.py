import h5py
import torch
from torch.utils.data import Dataset

class SegmentDataset(Dataset):
    def __init__(self, file_paths) -> None:
        super().__init__()
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, index):
        # 打开HDF5文件
        file_name = self.file_paths[index]
        with h5py.File(file_name, 'r') as file:
            # 加载原始视频数据，并将维度从 F×H×W×C 转换为 F×C×H×W
            appearance_data = file['raw_video'][:, :, :, -3:]
            appearance_data = appearance_data.transpose(0, 3, 1, 2)
            motion_data = file['raw_video'][:, :, :, :3]
            motion_data = motion_data.transpose(0, 3, 1, 2)
            
            # 加载标签数据
            label_data = file['preprocessed_label'][:]

            # 将Numpy数组转换为PyTorch张量
            appearance_data = torch.from_numpy(appearance_data).float()
            motion_data = torch.from_numpy(motion_data).float()
            label_data = torch.from_numpy(label_data).float()

        # 返回转置后的张量
        return torch.stack((appearance_data, motion_data)), label_data
