

import os
from typing import Optional, Sized

import numpy as np
import h5py
import torch
from torch.utils.data import ConcatDataset
from rppg.datasets.DeepPhysDataset import DeepPhysDataset
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader


def dataset_loader(fit_cfg, dataset_path):
    model_name = fit_cfg.model
    dataset_name = [fit_cfg.train.dataset_name, fit_cfg.test.dataset_name]
    time_length = fit_cfg.time_length
    batch_size = fit_cfg.train.batch_size
    overlap_interval = fit_cfg.overlap_interval
    img_size = fit_cfg.img_size
    larger_box_coef = fit_cfg.larger_box_coef
    train_flag = fit_cfg.train_flag
    eval_flag = fit_cfg.eval_flag
    debug_flag = fit_cfg.debug_flag

    save_root_path = dataset_path

    if dataset_name[0] == dataset_name[1]:
        root_file_path = save_root_path + dataset_name[0] + '_' + str(img_size) + '_' + str(larger_box_coef).replace('.', '_')
        path = get_all_files_in_path(root_file_path)
        path = path[:10]
        path_len = len(path)

        test_len = 0
        if eval_flag and train_flag:
            test_len = int(np.floor(path_len * 0.1))
            eval_path = path[-test_len:]
        else:
            eval_path = path

        if train_flag:
            train_len = int(np.floor((path_len) * 0.8))
            train_path = path[:train_len]
            val_path = path[train_len:]
    else:
        root_file_path = save_root_path + dataset_name[0] + '_' + str(img_size) + '_' + str(larger_box_coef).replace('.', '_')
        path = get_all_files_in_path(root_file_path)
        path_len = len(path)

        if train_flag:
            train_len = int(np.floor((path_len) * 0.9))
            train_path = path[:train_len]
            val_path = path[train_len:]
        if eval_flag:
            root_file_path = save_root_path + dataset_name[1] + '_' + str(img_size) + '_' + str(larger_box_coef).replace('.', '_')
            eval_path = get_all_files_in_path(root_file_path)
    
    dataset = []

    if train_flag:
        train_dataset = get_dataset(train_path, model_name, time_length, batch_size, overlap_interval, img_size)
        dataset.append(train_dataset)
        val_dataset = get_dataset(val_path, model_name, time_length, batch_size, overlap_interval, img_size)
        dataset.append(val_dataset)
    if eval_flag:
        eval_dataset = get_dataset(eval_path, model_name, time_length, batch_size, overlap_interval, img_size)
        dataset.append(eval_dataset)
    
    return dataset



def get_dataset(path, model_name, time_legth, batch_size, overlap_interval, img_size):
    idx = 0
    round_flag = 0
    rst_dataset = None
    datasets = []

    while True:
        print(idx, len(path), round_flag)
        if round_flag == 0:
            appearance_data = []
            motion_data = []
            label_data = []
            round_flag = 1
        elif round_flag == 1:
            if idx == len(path):
                break
            file_name = path[idx]
            idx += 1
            file = h5py.File(file_name, 'r')
            num_frame, w, h, c = file['raw_video'].shape
            appearance_data.extend(file['raw_video'][:, :, :, -3:])
            motion_data.extend(file['raw_video'][:, :, :, :3])

            if w != img_size or h != img_size:
                raise ValueError('Image size error')

            temp_label = file['preprocessed_label']
            if len(temp_label) != num_frame:
                raise ValueError('Label length error')
            label_data.extend(temp_label)

            num_frame = (num_frame // (time_legth * batch_size)) * (time_legth * batch_size)
            appearance_data = appearance_data[:num_frame]
            motion_data = motion_data[:num_frame]   
            label_data = label_data[:num_frame]

            file.close()
            round_flag = 2
        elif round_flag == 2:
            if model_name == "DeepPhys":
                dataset = DeepPhysDataset(appearance_data=np.asarray(appearance_data), 
                                          motion_data=np.asarray(motion_data), 
                                          target=np.asarray(label_data))
            else:
                raise NotImplementedError('Model not implemented (%s)' % model_name)
            datasets.append(dataset)
            rst_dataset = ConcatDataset(datasets)
            round_flag = 0
    return rst_dataset



def get_all_files_in_path(path):
    files = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            files.append(os.path.join(dirpath, filename))
    return files

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

def data_loader(datasets, fit_cfg):
    train_batch_size = fit_cfg.train.batch_size
    test_batch_size = fit_cfg.test.batch_size
    time_length = fit_cfg.time_length
    shuffle = fit_cfg.train.shuffle

    test_loader = []
    if len(datasets) == 3 or len(datasets) == 2:
        total_len_train = len(datasets[0])
        total_len_validation = len(datasets[1])
        idx_train = np.arange(total_len_train)
        idx_validation = np.arange(total_len_validation)
        if shuffle:
            idx_train = idx_train.reshape(-1, time_length)
            idx_train = np.random.permutation(idx_train)
            idx_train = idx_train.reshape(-1)
            idx_validation = idx_validation.reshape(-1, time_length)
            idx_validation = np.random.permutation(idx_validation)
            idx_validation = idx_validation.reshape(-1)
            shuffle = False
        sampler_train = ClipSamper(idx_train)
        sampler_validation = ClipSamper(idx_validation)

        train_loader = DataLoader(datasets[0], batch_size=train_batch_size * time_length, sampler=sampler_train, shuffle=shuffle, worker_init_fn=seed_worker, generator=g)
        validation_loader = DataLoader(datasets[1], batch_size=test_batch_size * time_length, sampler=sampler_validation, shuffle=shuffle, worker_init_fn=seed_worker, generator=g)

        if len(datasets) == 2:
            return [train_loader, validation_loader]
        if len(datasets) == 3:
            test_loader = DataLoader(datasets[2], batch_size=test_batch_size * time_length, shuffle=shuffle, worker_init_fn=seed_worker, generator=g)
            return [train_loader, validation_loader, test_loader]
    else:
        raise ValueError('Dataset length error')



class ClipSamper(Sampler):
    def __init__(self, data_source) -> None:
        self.data_source = data_source

    def __iter__(self):
        return iter(self.data_source.tolist())
    
    def __len__(self):
        return len(self.data_source.tolist())
