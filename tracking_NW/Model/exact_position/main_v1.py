from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow.contrib import rnn
from function import *
from inference import *
from load_data import *
from train import *
from testing import *
from rnn import *
import numpy as np
import skimage.measure
#State from ground truth, initial position:given, output location feed to input as mask

#1: Log files
fout_log= open("log.txt","w")
fout_log.write("TEST LOG PRINT\nSOO\n")

#2: Graph
#Training Parameters
validation_step=10;
learning_rate =0.001
X = tf.placeholder("float", [FLAGS.batch_size, None, feature_size*channels])
Y_state_in = tf.placeholder("float", [FLAGS.batch_size, None, 1])
Y_lonlat_out = tf.placeholder("float", [FLAGS.batch_size, None, 2])
timesteps = tf.shape(X)[1]
h=128; w=288

xx=Inference(X,timesteps)
prediction_lonlat, last_state = RNN(xx, weights, biases)
state=tf.concat([Y_state_in,Y_state_in],axis=2) 
state_sum=tf.reduce_sum(state)
prediction_lonlat=tf.multiply(prediction_lonlat,state) # prediction value when state prediction is 1
sqsum=tf.reduce_sum(tf.pow(prediction_lonlat - Y_lonlat_out,2))
loss_lonlat=tf.div(sqsum,state_sum) #Ave_MSE pre one element
loss_op=loss_lonlat*16000
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

#3 :Training
with tf.Session() as sess:
    # Initialize all variables
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    sess.run(init)
    path_test="/export/kim79/ECCV/heatmap_all/testset/"
    for rep in range(10000):
        #(1) seq_length = 8, iteration= 4 (epoch=8)
        path="/export/kim79/ECCV/heatmap_all/8/"
        iter=7
        for ii in range(iter):
            iteration=1
            train(sess,loss_op,train_op,X,Y_state_in,Y_lonlat_out,prediction_lonlat, last_state,iteration,path,fout_log);
            name=str(rep)+str(ii*iteration)+"_size8_"
            test(name,sess,loss_op,train_op,X,Y_state_in,Y_lonlat_out,prediction_lonlat,last_state,iteration,path_test,path);
        #(2) seq_length = 16, iteration= 2 (epoch=4)
        path="/export/kim79/ECCV/heatmap_all/16/"
        iteration=2
        train(sess,loss_op,train_op,X,Y_state_in,Y_lonlat_out,prediction_lonlat, last_state,iteration,path,fout_log);
        name=str(rep)+str(iteration)+"_size16_"
        test(name,sess,loss_op,train_op,X,Y_state_in,Y_lonlat_out,prediction_lonlat,last_state,iteration,path_test,path);
        #(3) seq_length = 32, iteration= 1 (epoch=2)
        path="/export/kim79/ECCV/heatmap_all/32/"
        iteration=1
        train(sess,loss_op,train_op,X,Y_state_in,Y_lonlat_out,prediction_lonlat, last_state,iteration,path,fout_log);
        name=str(rep)+str(iteration)+"_size32_"
        test(name,sess,loss_op,train_op,X,Y_state_in,Y_lonlat_out,prediction_lonlat,last_state,iteration,path_test,path);
        #(4) seq_length = 40, iteration= 1 (epoch=2)
        path="/export/kim79/ECCV/heatmap_all/40/"
        iteration=1
        train(sess,loss_op,train_op,X,Y_state_in,Y_lonlat_out,prediction_lonlat, last_state,iteration,path,fout_log);
        name=str(rep)+str(iteration)+"_size40_"
        test(name,sess,loss_op,train_op,X,Y_state_in,Y_lonlat_out,prediction_lonlat,last_state,iteration,path_test,path);
fout_log.close();
