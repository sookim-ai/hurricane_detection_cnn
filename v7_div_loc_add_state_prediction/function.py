# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import re
import sys
import tarfile
from six.moves import urllib
import tensorflow as tf
import numpy as np
parser = argparse.ArgumentParser()
# Basic model parameters.
parser.add_argument('--batch_size', type=int, default=24,
                    help='Number of images to process in a batch.')


parser.add_argument('--data_dir', type=str, default='/tmp/cifar10_data',
                    help='Path to the CIFAR-10 data directory.')

parser.add_argument('--use_fp16', type=bool, default=False,
                    help='Train the model using fp16.')

FLAGS = parser.parse_args()

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.
NUM_CLASSES = 4096 #cifar10_input.NUM_CLASSES
w = 129
h = 86
channels = 2

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

#need to check
def crop_around_lonlat(image,y_lonlat_in):
  """From large image,"image", crop sub region(10x10) centering (lon,lat)

  Args:
    image: X-[bachsize,1,feature_size(h*w)*channels]: (24,1,22188)
    y_lonlat_in: [batchsize,1,2] : (24,1,2)
  Returns:
    cropped_image: (24,10(h),10(w),channels) = (24,10,10,2)
  """
  image=np.reshape(image, [FLAGS.batch_size,1,h,w,channels])  
  cropped_image=[];
  for i in range(int(FLAGS.batch_size)):
      lon,lat=y_lonlat_in[i,0,:]
      lon_index=int(lon*w)
      lat_index=int(lat*h)
      lat_lb=lat_index-5
      lat_up=lat_index+5
      lon_lb=lon_index-5
      lon_up=lon_index+5
      if float(lat_index-5)<0.0 : 
          lat_lb=0 
          lat_up=lat_lb+10;
      if float(lat_index+5)>85.0:
          lat_up=85
          lat_lb=85-10
      if float(lon_index-5)<0.0:
          lon_lb=0
          lon_up=lon_lb+10
      if float(lon_index+5)>128.0:
          lon_up=128
          lon_lb=128-10
      cropped_image.append([image[i,0,lat_lb:lat_up,lon_lb:lon_up,:]])
  cropped_image=np.asarray(np.concatenate(cropped_image,axis=0))
  return cropped_image





def mask_around_lonlat(image_in,y_lonlat_in):
  """From large image,"image", crop sub region(10x10) centering (lon,lat)

  Args:
    image_in: X-[bachsize,timesteps,feature_size(h*w)*channels]: (24,timesteps,22188)
    y_lonlat_in: [batchsize,timesteps,2] : (24,timesteps,2)
  Returns:
    masked_image: (24,timesteps, 86(h)*129(w)*2(channels)): Mask size is 10x10
  """
  batch_size,timesteps,features=np.shape(image_in)
  mask=[];
  for i in range(int(FLAGS.batch_size)):
      mask_t=[];
      for t in range(int(timesteps)):
          image=image_in[i,t,:]
          image=np.reshape(image, [h,w,channels])
          lon,lat=y_lonlat_in[i,t,:]
          lon_index=int(lon*w)
          lat_index=int(lat*h)
          lat_lb=lat_index-3
          lat_up=lat_index+3
          lon_lb=lon_index-3
          lon_up=lon_index+3
          if float(lat_index-3)<0.0 :
              lat_lb=0
              lat_up=lat_lb+6;
          if float(lat_index+3)>85.0:
              lat_up=86
              lat_lb=86-6
          if float(lon_index-3)<0.0:
              lon_lb=0
              lon_up=lon_lb+6
          if float(lon_index+3)>128.0:
              lon_up=129
              lon_lb=129-6
          image[ 0:lat_lb,  :,  :]=0
          image[ lat_up:86, :,  :]=0
          image[   :,0:lon_lb,  :]=0
          image[   :,lon_up:129,:]=0
          mask_t.append(image)
      mask.append([mask_t])
  mask=np.asarray(np.concatenate(mask,axis=0))
  mask=np.reshape(mask,[FLAGS.batch_size,timesteps,h*w*channels])
  return mask


def div_of_lonlat(lonlat_in):
    """"
        input: lonlat_in: (batch_numbers, batch_size, timesteps, 2)
        output: lonlat_out: lonlat_in[i,j,t+1,:]/lonlat_in[i,j,t,:]
    """
    sh1,sh2,sh3,sh4=np.shape(lonlat_in)
    lonlat_out=np.zeros([sh1,sh2,sh3-1,sh4])
    for i in range(sh1):
        for j in range(sh2):
            for t in range(sh3-1):
                for k in range(sh4):
                    if (lonlat_in[i,j,t,k]==0):
                        lonlat_out[i,j,t,k]=0
                    else:
                        div=float(lonlat_in[i,j,t+1,k])/float(lonlat_in[i,j,t,k])    
                        lonlat_out[i,j,t,k]=div        
    return lonlat_out




def reconstruct_div_to_lonlat(div, initial_position):
    """"
        input: div = [batch_numbers,batch_size,timesteps,1,2(div_lon, div_lat)] 
               #(400,24,6,1,2)
               initial_position = [batch_numbers,batch_size,1(tstep),2] 
        output: lonlat: div[i,j,t,1,:]*lonlat[i,j,t-1,1,:]
    """
    sh1,sh2,sh3,sh4,sh5=np.shape(div) # sh3 is timesteps
    lonlat=np.zeros([sh1,sh2,sh3,sh4,sh5])
    for i in range(sh1):
        for j in range(sh2):
            for t in range(sh3):
                for k in range(sh5):
                    if t<1:
                        #Re-constructed value
                        value=float(div[i,j,t,0,k])*float(initial_position[i,j,0,k])         
                        lonlat[i,j,t,0,k]=value
                    else:
                        #Re-constructed value
                        value=float(div[i,j,t,0,k])*float(lonlat[i,j,t-1,0,k])              
                        lonlat[i,j,t,0,k]=value                     
    return lonlat


def reconstruct_one_lonlat(div,initial_position):
    """"
        input: div = [batch_size,1,2(div_lon, div_lat)] 
               initial_position = [batch_size,1(tstep),2] 
        output: lonlat: (24,1,2)
    """
    sh1,sh2,sh3=np.shape(div) #(24,1,2)
    div=np.reshape(div,[1,sh1,1,sh2,sh3])
    d1,d2,d3=np.shape(initial_position) #(24,1,2)
    initial_position=np.reshape(initial_position,[1,d1,d2,d3])
    lonlat=reconstruct_div_to_lonlat(div,initial_position)
    lonlat=np.reshape(lonlat,[sh1,sh2,sh3]) #(24,1,2)
    return lonlat


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var




def Inference(images,timesteps):
  """Build the CNN to embed climate image
  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 2, 32],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [32], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(pre_activation, name=scope.name)

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
  # norm1
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 32, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(pre_activation, name=scope.name)

  # norm2
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
  # pool2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  # local3
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
  # local4
  with tf.variable_scope('local4') as scope:
    weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)

  # linear layer(WX + b),
  # We don't apply softmax here because
  # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
  # and performs the softmax internally for efficiency.
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                          stddev=1/192.0, wd=0.0)
    biases = _variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    
    output_loc=tf.reshape(softmax_linear,[FLAGS.batch_size,1, -1])
  return output_loc



def inference_1layer(images):
  #conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 2, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(pre_activation, name=scope.name)

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
  # norm1
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')
  # local3
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool1, [FLAGS.batch_size*timesteps, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
  # local4
  with tf.variable_scope('local4') as scope:
    weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                          stddev=1/192.0, wd=0.0)
    biases = _variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    output_loc=tf.reshape(softmax_linear,[FLAGS.batch_size,timesteps, -1])
  return output_loc

def old_embedding(xx):
	#Make embedding of X using tf.nn.conv1d
	#temp=[];
	#for i in range(timesteps):
	# Convolution Layer with 32 filters and a kernel size of 5
	conv1 = tf.layers.conv2d(xx, 64, 5, activation=tf.nn.relu)
	# Max Pooling (down-sampling) with strides of 2 and kernel size of 2
	conv1 = tf.layers.max_pooling2d(conv1, 2, 2)
	# Convolution Layer with 64 filters and a kernel size of 5
	conv2 = tf.layers.conv2d(conv1, 64, 5, activation=tf.nn.relu)
	# Max Pooling (down-sampling) with strides of 2 and kernel size of 2
	conv2 = tf.layers.max_pooling2d(conv2, 2, 2)
	# Convolution Layer with 32 filters and a kernel size of 5
	conv3 = tf.layers.conv2d(conv2,32, 5, activation=tf.nn.relu)
	# Max Pooling (down-sampling) with strides of 2 and kernel size of 2
	conv3 = tf.layers.max_pooling2d(conv3, 2, 2)
	# Convolution Layer with 64 filters and a kernel size of 3
	conv4 = tf.layers.conv2d(conv3,32, 5, activation=tf.nn.relu)
	# Max Pooling (down-sampling) with strides of 2 and kernel size of 2
	conv4 = tf.layers.max_pooling2d(conv4, 2, 2)
	conv4_flatten=tf.reshape(conv4,[FLAGS.batch_size*timesteps,-1]);
	##FC
	fc1=tf.layers.dense(conv4_flatten,8192)
	fc1=tf.layers.dropout(fc1,rate=0.8);
	x_em=tf.layers.dense(fc1,4096)
	x_em=tf.reshape(x_em,[FLAGS.batch_size,timesteps,-1])
	print("SIZE  "+str(x_em));
        return x_em

def old_embedding_1layer(xx):
        #Make embedding of X using tf.nn.conv1d
        #temp=[];
        #for i in range(timesteps):
        # Convolution Layer with 32 filters and a kernel size of 5
        conv1 = tf.layers.conv2d(xx, 64, 5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)
        conv1_flatten=tf.reshape(conv1,[FLAGS.batch_size*timesteps,-1]);
        ##FC
        fc1=tf.layers.dense(conv1_flatten,8192)
        fc1=tf.layers.dropout(fc1,rate=0.8);
        x_em=tf.layers.dense(fc1,4096)
        x_em=tf.reshape(x_em,[FLAGS.batch_size,timesteps,-1])
        print("SIZE "+str(x_em));
        return x_em
