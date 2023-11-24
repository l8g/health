
import json
import math
import multiprocessing
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import sys

from tqdm import tqdm
sys.path.append('..')
from rppg.config import get_config
from utils.data_path import *
import os
import h5py
from utils.funcs import *
import cv2
import face_recognition

def process(cfg_path):
    cfg = get_config(cfg_path)
    preprocessing(cfg)
    return cfg


def preprocessing(cfg):
    print(f"dataset name: {cfg.preprocess.dataset_name}")
    print(f"data root path: {cfg.data_root_path}")
    print(f"dataset path: {cfg.dataset_path}")

    chunk_size = cfg.preprocess.process_num
    img_size = cfg.preprocess.image_size
    larger_box_coef = cfg.preprocess.larger_box_coef

    raw_data_path_loader = None
    if cfg.preprocess.dataset_name == 'UBFC':
        raw_data_path_loader = UBFC_RawDataPathLoader(cfg.data_root_path)
    elif cfg.preprocess.dataset_name == 'PURE':
        raw_data_path_loader = PURE_RawDataPathLoader(cfg.data_root_path)

    dataset_root_path = raw_data_path_loader.dataset_root_path
    data_list = raw_data_path_loader.data_list
    vid_name = raw_data_path_loader.video_name
    ground_truth_name = raw_data_path_loader.ppg_name




    chunk_num = math.ceil(len(data_list) / chunk_size)

    if chunk_num == 1:
        chunk_size = len(data_list)

    for chunk_idx in range(chunk_num):
        if chunk_idx == chunk_num - 1:
            chunk_data_list = data_list[chunk_idx * chunk_size:]
        else:
            chunk_data_list = data_list[chunk_idx * chunk_size: (chunk_idx + 1) * chunk_size]

        print(f'chunk_data_list: {chunk_data_list}')

        chunk_preprocessing(chunk_data_list, dataset_root_path, vid_name, ground_truth_name,
                            cfg.preprocess.dataset_name, cfg.dataset_path, img_size=img_size, larger_box_coef=larger_box_coef)




def chunk_preprocessing(data_list, dataset_root_path, vid_name, ground_truth_name, dataset_name, dataset_path, img_size, larger_box_coef):
    process = []
    save_root_path = dataset_path

    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    for data_path in data_list:
        proc = multiprocessing.Process(target=preprocess_data, args=(data_path, dataset_root_path, vid_name, ground_truth_name, dataset_name, save_root_path, img_size, larger_box_coef, return_dict))
        process.append(proc)
        proc.start()
    
    for proc in process:
        proc.join()
    
    manager.shutdown()



def preprocess_data(data_path, dataset_root_path, vid_name, ground_truth_name, dataset_name, save_root_path, img_size, larger_box_coef, return_dict):
    video_path = dataset_root_path + '/' + data_path + vid_name
    label_path = dataset_root_path + '/' + data_path + ground_truth_name

    raw_video, preprocessed_label, hrv = data_precess(video_path, label_path, img_size, larger_box_coef)
    if None in raw_video:
        return
    
    dir_path = save_root_path + '/' + dataset_name + '_' + str(img_size) + '_' + str(larger_box_coef).replace('.', '_')
    if not os.path.isdir(dir_path):
        mkdir_p(dir_path)

    data = h5py.File(dir_path + '/' + data_path + '.hdf5', 'w')
    data.create_dataset('raw_video', data=raw_video)
    data.create_dataset('preprocessed_label', data=preprocessed_label)
    data.create_dataset('hrv', data=hrv)
    data.close()
    
def mkdir_p(directory):
    if not directory:
        return
    if directory.endswith('/'):
        mkdir_p(directory[:-1])
    if os.path.isdir(directory):
        return
    mkdir_p(os.path.dirname(directory))
    os.mkdir(directory)
    

def data_precess(video_path, label_path, img_size, larger_box_coef):
    dection_model = 'hog' # 'hog' or 'cnn'
    xy_points = pd.DataFrame(columns=['bottom', 'right', 'top', 'left'])

    # for PURE dataset
    if video_path.__contains__('png'):
        path = video_path[:-4]
        data = sorted(os.listdir(path))[1:]
        frame_total = len(data)
        raw_label = get_label(label_path, frame_total)
        hrv = get_hrv_label(raw_label, fs = 60.) # 这里原版是30，但是我觉得应该是60

        for i in tqdm(range(frame_total), position=0, leave=True, desc=path):
            frame = cv2.imread(path + '/' + data[i])
            face_location = face_recognition.face_locations(frame, model=dection_model)
            if len(face_location) == 0:
                xy_points.loc[i] = [np.nan, np.nan, np.nan, np.nan]
            else:
                xy_points.loc[i] = face_location[0]
        
        valid_fr_idx = xy_points[xy_points['top'].notnull()].index.tolist()
        front_idx = valid_fr_idx[0]
        rear_idx = valid_fr_idx[-1]

        xy_points = xy_points[front_idx:rear_idx + 1]
        raw_label = raw_label[front_idx:rear_idx + 1]
        hrv = hrv[front_idx:rear_idx + 1]

        y_x_w = get_CntYX_Width(xy_points=xy_points, larger_box_coef=larger_box_coef)

        raw_video = np.empty((xy_points.__len__(), img_size, img_size, 3))
        for i, frame_num in enumerate(range(front_idx, rear_idx + 1)):
            frame = cv2.imread(path + '/' + data[frame_num])
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face = np.take(frame, np.arange(y_x_w[i][0] - y_x_w[i][2], y_x_w[i][0] + y_x_w[i][2]), axis=0, mode='clip')
            face = np.take(face, np.arange(y_x_w[i][1] - y_x_w[i][2], y_x_w[i][1] + y_x_w[i][2]), axis=1, mode='clip')
            face = (face / 255.).astype(np.float32)
            if img_size == y_x_w[i][2] * 2:
                raw_video[i] = face
            else:
                raw_video[i] = cv2.resize(face, (img_size, img_size), interpolation=cv2.INTER_AREA)
    else:
        cap = cv2.VideoCapture(video_path)
        frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        raw_label = get_label(label_path, frame_total)
        hrv = get_hrv_label(raw_label, fs = 30.)

        for i in tqdm(range(frame_total), position=0, leave=True, desc=video_path):
            ret, frame = cap.read()
            if not ret:
                break
            face_location = face_recognition.face_locations(frame, model=dection_model)
            if len(face_location) == 0:
                xy_points.loc[i] = [np.nan, np.nan, np.nan, np.nan]
            else:
                xy_points.loc[i] = face_location[0]
        cap.release()

        valid_fr_idx = xy_points[xy_points['top'].notnull()].index.tolist()
        front_idx = valid_fr_idx[0]
        rear_idx = valid_fr_idx[-1]

        xy_points = xy_points[front_idx:rear_idx + 1]
        raw_label = raw_label[front_idx:rear_idx + 1]
        hrv = hrv[front_idx:rear_idx + 1]

        y_x_w = get_CntYX_Width(xy_points=xy_points, larger_box_coef=larger_box_coef)

        cap = cv2.VideoCapture(video_path)
        for _ in range(front_idx):
            cap.read()
        
        raw_video = np.empty((xy_points.__len__(), img_size, img_size, 3), dtype=np.float32)
        for frame_num in range(rear_idx + 1):
            ret, frame = cap.read()
            if not ret:
                print(f"Can't receive frame : {video_path}")
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face = np.take(frame, range(y_x_w[frame_num][0] - y_x_w[frame_num][2], y_x_w[frame_num][0] + y_x_w[frame_num][2]), axis=0, mode='clip')
            face = np.take(face, range(y_x_w[frame_num][1] - y_x_w[frame_num][2], y_x_w[frame_num][1] + y_x_w[frame_num][2]), axis=1, mode='clip')
            face = (face / 255.).astype(np.float32)
            if img_size == y_x_w[frame_num][2] * 2:
                raw_video[frame_num] = face
            else:
                raw_video[frame_num] = cv2.resize(face, (img_size, img_size), interpolation=cv2.INTER_AREA)
        cap.release()
    
    raw_video = diff_normalize_video(raw_video)
    raw_label = diff_normalize_label(raw_label)

    return raw_video, raw_label, hrv

def diff_normalize_video(video_data):
    frame_total, h, w, c = video_data.shape
    raw_video = np.empty((frame_total - 1, h, w, 6), dtype=np.float32)
    padd = np.zeros((1, h, w, 6), dtype=np.float32)
    for frame_num in range(frame_total - 1):
        raw_video[:, :, :, :3] = generate_motion_difference(video_data[frame_num], video_data[frame_num + 1])
    raw_video[:, :, :, :3] = raw_video[:, :, :, :3] / np.std(raw_video[:, :, :, :3])
    raw_video = np.append(raw_video, padd, axis=0)
    video_data = video_data - np.mean(video_data)
    video_data = video_data / np.std(video_data)
    raw_video[:, :, :, 3:] = video_data
    raw_video[np.isnan(raw_video)] = 0
    return raw_video

def generate_motion_difference(prev_frame, crop_frame):
    motion_input = (crop_frame - prev_frame) / (crop_frame + prev_frame + 1e-8)
    return motion_input


            
def diff_normalize_label(label):
    delta_label = np.diff(label, axis=0)
    delta_label /= np.std(delta_label)
    delta_label = np.array(delta_label).astype(np.float32)  
    delta_label = np.append(delta_label, np.zeros(1, dtype=np.float32), axis=0)
    delta_label[np.isnan(delta_label)] = 0
    return delta_label


def get_CntYX_Width(xy_points, larger_box_coef):
    y_range_ext = (xy_points.top - xy_points.bottom) * 0.2 # for forehead
    xy_points.bottom = xy_points.bottom - y_range_ext
    xy_points['cnt_y'] = (xy_points.top + xy_points.bottom) / 2
    xy_points['cnt_x'] = (xy_points.right + xy_points.left) / 2
    xy_points['bbox_half_size'] = (xy_points.top - xy_points.bottom).median() * larger_box_coef / 2

    xy_points = xy_points.interpolate()

    xy_points.cnt_x = xy_points.cnt_x.ewm(alpha=0.1).mean()
    xy_points.cnt_y = xy_points.cnt_y.ewm(alpha=0.1).mean()
    xy_points.bbox_half_size = xy_points.bbox_half_size.ewm(alpha=0.1).mean()

    xy_points = xy_points.round().astype(int)
    return xy_points[['cnt_y', 'cnt_x', 'bbox_half_size']].values


def get_label(label_path, frame_total):
    if label_path.__contains__('json'):
        name = label_path.split('/')
        label = []
        label_time = []
        label_hr = []
        time = []

        with open(label_path[:-4] + name[-2] + '.json', 'r') as json_file:
            json_data = json.load(json_file)
            for data in json_data['/FullPackage']:
                label.append(data['Value']['waveform'])
                label_time.append(data['Timestamp'])
                label_hr.append(data['Value']['pulseRate'])
            for data in json_data['/Image']:
                time.append(data['Timestamp'])
    else:
        f = open(label_path, 'r')
        f_read = f.read().split('\n')
        label = ' '.join(f_read[0].split()).split()
        label_hr = ' '.join(f_read[1].split()).split()
        label = list(map(float, label))
        label = np.array(label).astype(np.float32)
        label_hr = list(map(float, label_hr))
        label_hr = np.array(label_hr).astype('int')
        f.close()
    
    label = list(map(float, label))
    if len(label) != frame_total:
        label = np.interp(
            np.linspace(1, len(label), frame_total), 
            np.linspace(1, len(label), len(label)), 
            label)
        
    label = np.array(label).astype(np.float32)
    return label

def get_hrv_label(ppg_signal, fs = 30.):
    clean_ppg = detrend(ppg_signal, 100)
    clean_ppg = BPF(clean_ppg, fs = fs)
    hrv = get_hrv(clean_ppg, fs = fs)
    return hrv.astype(np.float32)

                
def test_preprecess_ubfc_subject():
    dataset_root_path = '/mnt/e/UBFC_DATASET/UBFC'
    vid_name = '/vid.avi'
    ground_truth_name = '/ground_truth.txt'
    dataset_name = 'UBFC'
    save_root_path = '/mnt/e/preprocessed/'
    img_size = 64
    larger_box_coef = 1.5
    return_dict = None
    data_path = 'subject1'
    preprocess_data(data_path, dataset_root_path, vid_name, ground_truth_name, dataset_name, save_root_path, img_size, larger_box_coef, return_dict)

def test_preprocess_pure_subject():
    dataset_root_path = '/mnt/e/UBFC_DATASET/PURE'
    vid_name = '/png'
    ground_truth_name = '/json'
    dataset_name = 'PURE'
    save_root_path = '/mnt/e/preprocessed/'
    img_size = 64
    larger_box_coef = 1.5
    return_dict = None
    data_path = '01-01'
    preprocess_data(data_path, dataset_root_path, vid_name, ground_truth_name, dataset_name, save_root_path, img_size, larger_box_coef, return_dict)

    
if __name__ == '__main__':
    cfg_path = '../configs/base_config.yaml'
    cfg = process(cfg_path)
    print('preprocess done!')

    # test_preprecess_ubfc_subject()
    # test_preprocess_pure_subject()
