

import os
import numpy as np
import h5py


def split_segment(file_path : str, out_path : str, time_length : int):
    """
    Split the video into segments with the given time length.
    """
    # Open the hdf5 file
    file = h5py.File(file_path, 'r')
    # Get the length of the video
    data_length = file['raw_video'].shape[0]
    # Calculate the number of segments
    num_segments = data_length // time_length

    out_path = os.path.join(out_path, os.path.basename(os.path.dirname(file_path)) + '_' + str(time_length))
    os.makedirs(out_path, exist_ok=True)
    
    for i in range(num_segments):
        start_idx = i * time_length
        end_idx = (i + 1) * time_length
        # Read and store each segment
        video_segments = file['raw_video'][start_idx:end_idx]
        label_segments = file['preprocessed_label'][start_idx:end_idx]
        hrv_segments = file['hrv'][start_idx:end_idx]

        # Write the segments to the output file
        output_file_path = os.path.join(out_path, os.path.basename(file_path)[:-5] + '_' + str(i) + '.hdf5')
        with h5py.File(output_file_path, 'w') as output_file:
            output_file.create_dataset('raw_video', data=video_segments)
            output_file.create_dataset('preprocessed_label', data=label_segments)
            output_file.create_dataset('hrv', data=hrv_segments)
    
    file.close()


def split_all_files(input_dir : str, out_dir : str, time_length : int):
    """
    Split all the files in the input directory into segments with the given time length.
    """
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.hdf5'):
            file_path = os.path.join(input_dir, file_name)
            split_segment(file_path, out_dir, time_length)


if __name__ == '__main__':
    input_dir = '/mnt/e/preprocessed/UBFC_128_1_5'
    out_dir = '/mnt/e/preprocessed/segments'
    time_length = 180
    split_all_files(input_dir, out_dir, time_length)