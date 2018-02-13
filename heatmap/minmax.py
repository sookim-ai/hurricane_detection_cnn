
import numpy as np
from tqdm import tqdm

n_chan = 2

if n_chan == 2:
    data_dir = 'detection_cnn_lr_2ch/heatmap_data/'
    img = np.load(data_dir + 'hurricane_heatmap_input_2channels.npy', mmap_mode='r')
    running_max = [-np.inf, -np.inf]
    running_min = [np.inf, np.inf]
elif n_chan == 3:
    data_dir = 'detection_cnn_hr_3ch/heatmap_data/'
    img = np.load(data_dir + 'hurricane_heatmap_input_3channels.npy', mmap_mode='r')
    running_max = [-np.inf, -np.inf, -np.inf]
    running_min = [np.inf, np.inf, np.inf]
else:
    raise ValueError


print('Calculating min-max of each channels')
print(data_dir)

d1, d2, d3, d4 = img.shape
for i in tqdm(range(d1)):
    this_img = img[i,:,:,:]
    max_vec = this_img.max(axis=1).max(axis=0)
    min_vec = this_img.min(axis=1).min(axis=0)

    running_max = np.maximum(running_max, max_vec)
    running_min = np.minimum(running_min, min_vec)

    if i % 500 == 0:
        print(running_max)
        print(running_min)

print(running_max)
print(running_min)
np.save(data_dir + 'max.npy', running_max)
np.save(data_dir + 'min.npy', running_min)


