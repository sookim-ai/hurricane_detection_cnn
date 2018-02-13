"""
Usage: 3_pick_centers.py DATASET MODEL [options]

Options:
    -h, --help        show this help message
    --test            test
"""

import os, sys
from tqdm import tqdm
import numpy as np
from docopt import docopt
from skimage import measure
import pandas as pd

args = docopt(__doc__)
data_select = args['DATASET']
model_name = args['MODEL']
b_test = args['--test']
threshold = 0.8


try:
    assert data_select in ('2ch', '3ch')
    print('### Training model for {}'.format(data_select))
except:
    raise ValueError('The argument of script should be either "2ch" or "3ch". Got {}'.format(data_select))

"""setup path for each data"""
if data_select == '2ch':
    data_dir = 'detection_cnn_lr_2ch'
    model_dir = 'model'
    n_input_chan = 2
elif data_select == '3ch':
    data_dir = 'detection_cnn_hr_3ch'
    model_dir = 'model'
    n_input_chan = 3

model_path = os.path.join(data_dir, model_dir, model_name)
if b_test:
    heatmap_file = os.path.join(model_path, model_name + '_heatmap_test.npy')
else:
    heatmap_file = os.path.join(model_path, model_name + '_heatmap.npy')

all_heatmap = np.load(heatmap_file, mmap_mode='r')
n_heatmap, h_size, w_size = all_heatmap.shape
WW, HH = np.meshgrid(np.arange(w_size), np.arange(h_size))  # index matrix

l_result = []
for heatmap_idx in range(n_heatmap):
    heatmap = all_heatmap[heatmap_idx, :,:]
    thresholded = (heatmap > 0.8).astype('int')
    conn, num = measure.label(thresholded, connectivity=2, return_num=True)

    for component_idx in range(num):
        component_h = HH[conn==component_idx]
        component_w = WW[conn==component_idx]
        probs = heatmap[conn==component_idx]
        max_idx = np.argmax(probs)
        center_h = component_h[max_idx]
        center_w = component_w[max_idx]
        #print(center_h, center_w)
        l_result.append((heatmap_idx, center_h, center_w))

df_result = pd.DataFrame(l_result, columns=['frame', 'lon_idx', 'lat_idx'])
if b_test:
    result_file_name = model_name + '_center_test.csv'
else:
    result_file_name = model_name + '_center.csv'
print(result_file_name)

df_result.to_csv(result_file_name)
