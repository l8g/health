
import h5py
from torch.utils.data import Dataset

from dataset_loader import get_all_files_in_path
class UBFC_dataset(Dataset):
    def __init__(self, dataset_root, image_size, larger_box_coef, transform, target_transform) -> None:
        super().__init__()
        self.dataset_path = dataset_root + 'UBFC_' + str(image_size) + '_' + str(larger_box_coef).replace('.', '_')
        self.dataset = get_all_files_in_path(self.dataset_path)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        file_name = self.dataset[index]
        file = h5py.File(file_name, 'r')
        appearance_data = file['raw_video'][:, :, :, -3:]
        motion_data = file['raw_video'][:, :, :, :3]
        label_data = file['preprocessed_label']
        file.close()
        if self.transform:
            appearance_data = self.transform(appearance_data)
            motion_data = self.transform(motion_data)
        if self.target_transform:
            label_data = self.target_transform(label_data)
        return appearance_data, motion_data, label_data


        
