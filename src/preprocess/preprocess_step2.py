import os
import h5py
import numpy as np

def process_file_segments(input_path, segment_length, output_file, dataset_counter):
    with h5py.File(input_path, 'r') as input_file:
        # 假设所有数据集的第一个维度是时间或顺序
        data_length = input_file['raw_video'].shape[0]
        # 计算分段数量
        num_segments = data_length // segment_length
        
        # 初始化临时数组，用于存储片段
        video_segments = np.empty((num_segments, segment_length, *input_file['raw_video'].shape[1:]), dtype=input_file['raw_video'].dtype)
        label_segments = np.empty((num_segments, segment_length, *input_file['preprocessed_label'].shape[1:]), dtype=input_file['preprocessed_label'].dtype)
        hrv_segments = np.empty((num_segments, segment_length), dtype=input_file['hrv'].dtype)

        for i in range(num_segments):
            start_idx = i * segment_length
            end_idx = (i + 1) * segment_length
            # 读取并存储每个片段
            video_segments[i] = input_file['raw_video'][start_idx:end_idx]
            label_segments[i] = input_file['preprocessed_label'][start_idx:end_idx]
            hrv_segments[i] = input_file['hrv'][start_idx:end_idx]

        # 写入片段到输出文件
        if dataset_counter == 0:
            # 创建数据集
            output_file.create_dataset('raw_video', data=video_segments, maxshape=(None, segment_length, *input_file['raw_video'].shape[1:]), chunks=True)
            output_file.create_dataset('preprocessed_label', data=label_segments, maxshape=(None, segment_length, *input_file['preprocessed_label'].shape[1:]), chunks=True)
            output_file.create_dataset('hrv', data=hrv_segments, maxshape=(None, segment_length), chunks=True)
        else:
            # 扩展数据集
            output_file['raw_video'].resize((output_file['raw_video'].shape[0] + num_segments), axis=0)
            output_file['raw_video'][-num_segments:] = video_segments
            output_file['preprocessed_label'].resize((output_file['preprocessed_label'].shape[0] + num_segments), axis=0)
            output_file['preprocessed_label'][-num_segments:] = label_segments
            output_file['hrv'].resize((output_file['hrv'].shape[0] + num_segments), axis=0)
            output_file['hrv'][-num_segments:] = hrv_segments
        
        return num_segments

def process_all_files(input_dir, output_path, segment_length):
    dataset_counter = 0
    segment_counts = {}
    
    with h5py.File(output_path, 'w') as output_file:
        # 遍历目录下的所有.hdf5文件
        for file_name in os.listdir(input_dir):
            if file_name.endswith('.hdf5'):
                file_path = os.path.join(input_dir, file_name)
                # 处理单个文件并更新计数器
                num_segments = process_file_segments(file_path, segment_length, output_file, dataset_counter)
                dataset_counter += num_segments
                # 记录片段数量
                segment_counts[file_name] = num_segments

        # 创建记录片段数量的数据集
        output_file.create_dataset('segment_counts', data=np.array(list(segment_counts.items()), dtype=h5py.special_dtype(vlen=str)))

# 使用示例
input_dir = 'your/input/directory'
output_path = 'your/output/directory/combined_data.hdf5'
time_length = 180  # 片段长度
process_all_files(input_dir, output_path, time_length)
