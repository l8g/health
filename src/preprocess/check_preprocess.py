import h5py
import matplotlib.pyplot as plt
import numpy as np

path = '/mnt/e/preprocessed/UBFC641_5/subject1.hdf5'

with h5py.File(path, 'r') as f:
    raw_video = f['raw_video'][:]
    preprocessed_label = f['preprocessed_label'][:]
    hrv = f['hrv'][:]

print(raw_video.shape)
print(preprocessed_label.shape)



# 选择第一帧
frame = raw_video[10]

# # 提取前三个通道
# channel_1 = frame[:, :, 3]
# channel_2 = frame[:, :, 4]
# channel_3 = frame[:, :, 5]

# # 绘图
# plt.figure(figsize=(15, 5))

# plt.subplot(1, 3, 1)
# plt.imshow(channel_1, cmap='gray')
# plt.title('Channel 1')
# plt.axis('off')

# plt.subplot(1, 3, 2)
# plt.imshow(channel_2, cmap='gray')
# plt.title('Channel 2')
# plt.axis('off')

# plt.subplot(1, 3, 3)
# plt.imshow(channel_3, cmap='gray')
# plt.title('Channel 3')
# plt.axis('off')

# plt.show()


# 提取并组合前三个通道
rgb_frame = frame[:, :, :3]

# 找到最小和最大值，并进行缩放
min_val = np.min(rgb_frame)
max_val = np.max(rgb_frame)

normalized_frame = (rgb_frame - min_val) / (max_val - min_val)

# 绘图
plt.figure(figsize=(5, 5))
plt.imshow(normalized_frame)
plt.title('RGB Frame')
plt.axis('off')
plt.show()