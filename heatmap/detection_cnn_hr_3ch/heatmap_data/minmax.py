
import numpy as np
from tqdm import tqdm

img = np.load('hurricane_heatmap_input_3channels.npy', mmap_mode='r')

print('clipping the first dimension')
img_clipped = img.copy()
img_clipped[:,:,:,0] = np.clip(img[:,:,:,0], 0., 10.)
clipped_file = 'hurricane_heatmap_input_3channels_clip.npy'
np.save(clipped_file, img_clipped)

print('saved! ', clipped_file)
img = img_clipped

d1, d2, d3, d4 = img.shape
running_max = [-np.inf, -np.inf, -np.inf]
running_min = [np.inf, np.inf, np.inf]
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
np.save('max_clip.npy', running_max)
np.save('min_clip.npy', running_min)


