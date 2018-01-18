"""
Usage: train_detection_cnn.py DATA [options]

Options:
    -h, --help        show this help message
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

#Convolutional Network--------
def build_detection_CNN(input_placeholder, output_placeholder):
    """build a computational graph for the detection CNN.
    Parameters
    ----------
    input_placeholder: tf.placeholder
    input_dim: 
    """
    def weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

    def bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)

    def conv2d(x_in, W, padding='SAME'):
      return tf.nn.conv2d(x_in, W, strides=[1, 1, 1, 1], padding=padding)

    def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')

    # get shapes of input and output 
    _, xd2, xd3, n_input_chan = input_placeholder.get_shape()
    _, label_size = output_placeholder.get_shape()
    xd2, xd3, n_input_chan, label_size = map(int, [xd2, xd3, n_input_chan, label_size])

    h_pool2 = []
    for i in range(n_input_chan):
         # (0) slice corresponding channel
         x_image = x[:,:,:,i:i+1]
         # (1) First Convolutional Layer
         W_conv1 = weight_variable([5, 5, 1, 32])
         b_conv1 = bias_variable([32])
         conv=tf.nn.conv2d(x_image, W_conv1,strides=[1, 1, 1, 1], padding='SAME')
         h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1)+b_conv1)    
         h_pool1 = max_pool_2x2(h_conv1)
        # (2) Second Convolutional Layer
         W_conv2 = weight_variable([7, 7, 32, 64])
         b_conv2 = bias_variable([64])
         h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
         h_pool2.append(max_pool_2x2(h_conv2))

    # (3) Merge chanels  
    h_pool2_concat = tf.concat(h_pool2, 3)
    pd1, pd2, pd3, pd4 = h_pool2_concat.get_shape() 

    # ------- CONV as FC ----------------
    # (4) FC Layer 1
    W_fc1 = weight_variable([int(pd2), int(pd3), int(pd4), 128]) 
    b_fc1 = bias_variable([128])
    h_fc1 = tf.nn.relu(conv2d(h_pool2_concat, W_fc1, padding='VALID') + b_fc1)

    # (5) Dropout 
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # (6) FC Layer 2
    hd1, hd2, hd3, hd4 = h_fc1_drop.get_shape()
    W_fc2 = weight_variable([int(hd2), int(hd3), int(hd4), label_size]) 
    b_fc2 = bias_variable([label_size]) 
    y_conv = tf.nn.sigmoid(conv2d(h_fc1_drop, W_fc2, padding='VALID') + b_fc2)
    y_conv_squeezed = tf.squeeze(y_conv, axis=[1, 2])

    tf.add_to_collection('pred_prob', y_conv)
    tf.add_to_collection('keep_prob', keep_prob)
                      
    #(6) Train Variable
    cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y_conv_squeezed))
    learning_rate = 1e-5
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.cast(tf.greater(y_conv_squeezed, 0.5), tf.float32), y_)
    prediction = y_conv; 
    answer = y_;
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # return a dict of important tensors
    output_tensors = {'input_placeholder': x,
                      'label_placeholder': y_,
                      'keep_prob': keep_prob,
                      'pred_y_map': y_conv,
                      'pred_y_squeezed': y_conv_squeezed,
                      'cross_entropy': cross_entropy,
                      'train_op': train_op,
                      'accuracy': accuracy}
    return output_tensors


if __name__ == '__main__':
    """select input data"""
    args = docopt(__doc__)
    data_select = args['DATA']
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

    path = os.path.join(data_dir, "data")
    model_path = os.path.join(data_dir, model_dir)

    #output log
    fo = open("{}_out_log.txt".format(data_select), 'w')

    # Read Data
    batch_xs_in = np.load(os.path.join(path, "hurricane_input.npy"))
    batch_ys_in = np.load(os.path.join(path, "hurricane_label.npy"))
    xd1, xd2, xd3, xd4 = np.shape(batch_xs_in) #(75256, 40, 40, 3)
    yd1, yd2 = np.shape(batch_ys_in) #(75256, 1)
    channels = xd4

    print("Size of Data: "+str(xd2)+" by "+str(xd3)+" "+str(xd4)+"channels\n")
    print("Size of label: "+str(yd1)+" by "+str(yd2)+"\n")

    def shuffle_tensors(batch_xs_in,batch_ys_in):
        index_list=[i for i in range(yd1)];
        random.shuffle(index_list)
        batch_xs=[]; batch_ys=[];
        for i in index_list:
            batch_xs.append(batch_xs_in[i,:,:,:])
            batch_ys.append(batch_ys_in[i,:])
        return batch_xs,batch_ys

    #shuffle, 90% trainning 10% testing
    batch_xs, batch_ys = shuffle_tensors(batch_xs_in, batch_ys_in)
    num_tr = int(xd1*0.9)
    training_data = np.asarray(batch_xs[0:num_tr])
    test_data = np.asarray(batch_xs[num_tr:xd1])
    training_label = np.asarray(batch_ys[0:num_tr]) 
    test_label = np.asarray(batch_ys[num_tr:xd1])

    print(np.shape(training_data), np.shape(test_data),np.shape(training_label), np.shape(test_label))

    n_train = training_data.shape[0]
    batch_size = 50
    n_batch = math.ceil(n_train / batch_size)
    n_epoch = 1000
    last_batch_idx = n_batch - 1  # for checking the last trailing batch

    print('# Num training data: {}'.format(n_train))
    print('batch_size: {}'.format(batch_size))
    print('the number of batchs: {}'.format(n_batch))
    print('the number of epochs: {}'.format(n_epoch))

    # Build the computational graph
    x = tf.placeholder(tf.float32, [None, xd2, xd3, n_input_chan], name='input_tensor')
    y_ = tf.placeholder(tf.float32, [None,yd2], name='label_tensor')      
    output_tensors = build_detection_CNN(x, y_)
    train_op = output_tensors['train_op']
    keep_prob = output_tensors['keep_prob']
    accuracy = output_tensors['accuracy']
    cross_entropy = output_tensors['cross_entropy']

    #(7) Session Start: Train
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    try:
        for i_epoch in range(n_epoch):
            l_running_acc = []
            l_running_loss = []
            # iterate over bathes
            for i_batch in tqdm(range(n_batch)):
                # Construct Training Batch
                bch_start = i_batch * batch_size
                if i_batch != last_batch_idx:
                    bch_end = (i_batch + 1) * batch_size
                else:  # upto the last data even though the size of batch is smaller than batch_size
                    bch_end = n_train

                batch_x = training_data[bch_start:bch_end]
                batch_y = training_label[bch_start:bch_end]

                _, train_acc, train_loss = sess.run([train_op, accuracy, cross_entropy], feed_dict={x:batch_x, y_: batch_y, keep_prob: 0.5})
                l_running_acc.append(train_acc)
                l_running_loss.append(train_loss)
            train_accuracy = np.mean(l_running_acc)
            train_loss = np.mean(l_running_loss)

            # Print Training Accuracy -- not accurate
            if i_epoch % 1 == 0:
                print(" %dth Training Step, training accuracy %g, training loss %g"%(i_epoch, train_accuracy, train_loss))
                fo.write(" %dth Training Step, training accuracy %g"%(i_epoch, train_accuracy))

            #(8)Print Acuuracy
            if i_epoch % 1 == 0:
                print("\t %d : test accuracy %g"%(i_epoch,sess.run(accuracy, feed_dict={ 
                     x:test_data,  y_:test_label, keep_prob: 1.0})))
                fo.write("\t %d : test accuracy %g \n"%(i_epoch,sess.run(accuracy, feed_dict={
                     x:test_data, y_:test_label, keep_prob: 1.0})))
    except KeyboardInterrupt:
        print("KeyboardInterrupt accepted: Saving current model...")
        print("\t %d : test accuracy %g"%(i_epoch, sess.run(accuracy, feed_dict={ 
             x:test_data,  y_:test_label, keep_prob: 1.0})))

    fo.close()
    # Save the variables to disk.
    save_path = saver.save(sess, os.path.join(model_path, "detection_cnn.ckpt"))
    print("Model saved in file: %s" % save_path)
    #
