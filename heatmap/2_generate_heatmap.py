"""
Usage: 2_generate_heatmap.py DATASET MODEL [options]

Options:
    -h, --help        show this help message
    --test            for testing
"""


import os,sys
from math import *
import tensorflow as tf
import numpy as np
import csv
import random
from docopt import docopt
from tqdm import tqdm
import math
import pickle 
import time
import h5py


args = docopt(__doc__)
data_select = args['DATASET']
model_name = args['MODEL']
b_test = args['--test']
threshold = 0.8

#
# path configure
#

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
ckpt_file = os.path.join(model_path, model_name)
print('ckpt_file: ', ckpt_file)

#
# load data
#


whole_img_path = os.path.join(data_dir, 'heatmap_data','hurricane_heatmap_input_{}channels.npy'.format(n_input_chan))
whole_img = np.load(whole_img_path, mmap_mode='r')
_, imgd2, imgd3, imgd4 = whole_img.shape
whole_img_max = np.load(os.path.join(data_dir, 'heatmap_data', 'max.npy'))
whole_img_min = np.load(os.path.join(data_dir, 'heatmap_data', 'min.npy'))

#clip_max = 100
#whole_img_max[0] = clip_max 
print(whole_img_max)
print(whole_img_min)

#
# load model
#


saver = tf.train.import_meta_graph(ckpt_file+'.meta')
sess = tf.Session()
saver.restore(sess, ckpt_file)

'''placeholders for conv'''
img_input = tf.get_collection('img_input')[0]
pred_y_map = tf.get_collection('pred_y_map')[0]
keep_prob = tf.get_collection('keep_prob')[0]

'''placeholders for sliding window'''
input_x = tf.get_collection('crop_input')[0]
pred_y = tf.get_collection('crop_pred')[0]


#
# generate heatmap
#

n_frame, n_lat, n_lon, _ = whole_img.shape
_, md2, md3, _ = pred_y_map.shape.as_list()  # model output dimensions
d2_offset = int((n_lat - md2) / 2.)
d3_offset = int((n_lon - md3) / 2.)
print('offsets: d2 {}, d3{}'.format(d2_offset, d3_offset))
time_s = time.time()

n_heatmap_batch_size = 5 
n_heatmap_batch = math.ceil(n_frame / n_heatmap_batch_size)
last_batch_idx = n_heatmap_batch - 1

result_map = np.zeros((n_frame, n_lat, n_lon))
for batch_idx in tqdm(range(n_heatmap_batch)):
    b_start = batch_idx * n_heatmap_batch_size
    if batch_idx == last_batch_idx:
        b_end = n_frame
    else:
        b_end = (batch_idx+1) * n_heatmap_batch_size
    if b_test:
        b_start = 777
        b_end = 778
    frame = whole_img[b_start:b_end, :, :, :].copy()
    # frame[:,:,:,0] = np.clip(frame[:,:,:,0], 0., clip_max)
    
    frame = (frame - whole_img_min) / (whole_img_max - whole_img_min)
    
    out = sess.run(pred_y_map, feed_dict={img_input: frame,
                                      keep_prob: 1.0})
    result_map[b_start:b_end, d2_offset:d2_offset+md2, d3_offset:d3_offset+md3] = out[:,:,:,0]
    
    if b_test:
        print('break due to test')
        break

if b_test:
    heatmap_file = ckpt_file + '_heatmap_test.npy'
    #np.save(heatmap_file, result_map[:batch_idx*n_heatmap_batch_size])
    np.save(heatmap_file, result_map[b_start:b_end])
    print(heatmap_file)
else:
    heatmap_file = ckpt_file + '_heatmap.npy'
    print(heatmap_file)
    time_s = time.time()
    np.save(heatmap_file, result_map)
    #f = h5py.File(heatmap_file, 'w')
    #f.create_dataset('heatmap', data=result_map)
    print('time to save file: {:.2f}sec'.format(time.time() - time_s))



