#!/usr/bin/env python
import os,sys
from math import *
import tensorflow as tf
import numpy as np
import csv
import random
import skimage.measure
import copy
#output log
fo=open("out_log.txt",'w')
# Read Data
#path="/export/kim79/h2/finished_npy/7/hurricane_crop_lr2ch/old_dataset_16upscale/"
path="/export/kim79/h2/finished_npy/7/hurricane_crop_hr2ch/"
xs_in=np.load(path+"hurricane_input.npy")
ys_in=np.load(path+"hurricane_label.npy")
batch_xs_in=skimage.measure.block_reduce(xs_in, (1,4,4,1), np.max);
batch_ys_in=ys_in

channels=2
xd1,xd2,xd3,xd4=np.shape(batch_xs_in) #(75256, 10, 10, 3)
yd1,yd2=np.shape(batch_ys_in) #(75256, 2)

print("Size of Data: "+str(xd1)+" by "+str(xd2)+" by "+str(xd3)+" "+str(xd4)+"channels\n")
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
batch_xs,batch_ys=shuffle_tensors(batch_xs_in, batch_ys_in)
num_tr=int(xd1*0.9)
training_data=np.asarray(batch_xs[0:num_tr])
test_data=np.asarray(batch_xs[num_tr:xd1])
training_label=np.asarray(batch_ys[0:num_tr]) 
test_label=np.asarray(batch_ys[num_tr:xd1])

print(np.shape(training_data), np.shape(test_data),np.shape(training_label), np.shape(test_label))


#Convolutional Network--------
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x_in, W):
  return tf.nn.conv2d(x_in, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

#START
W_conv1=[0,0,0];W_conv2=[0,0,0];W_conv3=[0,0,0];b_conv1=[0,0,0];b_conv2=[0,0,0];b_conv3=[0,0,0];h_conv1=[0,0,0];h_pool1=[0,0,0];h_conv2=[0,0,0];h_pool2=[0,0,0];h_conv3=[0,0,0];h_pool3=[0,0,0];pre=[0,0,0];
W_conv1_2=[0,0];W_conv2_2=[0,0,0];W_conv3_2=[0,0,0];b_conv1_2=[0,0,0];b_conv2_2=[0,0,0];b_conv3_2=[0,0,0];h_conv1_2=[0,0,0];h_conv2_2=[0,0,0];h_conv3_2=[0,0,0];
label_size=2;

#Define In/Output seperately: List is unhashable
x = [tf.placeholder(tf.float32, [None,xd2,xd3]) for i in range(channels)]
y_ = tf.placeholder(tf.float32, [None,yd2]);      
for i in range(channels):
    #(1) First Convolutional Layer
     W_conv1[i] = weight_variable([3, 3, 1, 32])
     b_conv1[i] = bias_variable([32])
     x_image=tf.reshape(x[i],[-1,10,10,1])
     print(x_image.get_shape(),W_conv1[i].get_shape())
     conv=tf.nn.conv2d(x_image, W_conv1[i],strides=[1, 1, 1, 1], padding='SAME')
     print(x_image.get_shape(),W_conv1[i].get_shape(),conv.get_shape(),b_conv1[i].get_shape())
     h_conv1[i] = tf.nn.relu(conv2d(x_image, W_conv1[i])+b_conv1[i])    
     h_pool1[i] = max_pool_2x2(h_conv1[i])
    #(2) Second Convolutional Layer
     W_conv2[i] = weight_variable([3, 3, 32, 64])
     b_conv2[i] = bias_variable([64])
     h_conv2[i] = tf.nn.relu(conv2d(h_pool1[i], W_conv2[i]) + b_conv2[i])
     h_pool2[i] = max_pool_2x2(h_conv2[i])
    #(3) Third Convolutional Layer
#     W_conv3[i] = weight_variable([5, 5, 64, 128])
#     b_conv3[i] = bias_variable([128])
#     h_conv3[i] = tf.nn.relu(conv2d(h_pool2[i], W_conv3[i]) + b_conv3[i])
#     h_pool3[i] = max_pool_2x2(h_conv3[i])
     
#(3) Densely Connected Layer
h_pool2_concat=tf.concat([h_pool2[0],h_pool2[1]],3)
pd1,pd2,pd3,pd4=h_pool2_concat.get_shape() 
print(pd1,pd2,pd3,pd4)
W_fc1 = weight_variable([int(pd2)*int(pd3)*int(pd4), 64]) 
b_fc1 = bias_variable([64])
h_pool2_flat = tf.reshape(h_pool2_concat, [-1,int(pd2)*int(pd3)*int(pd4)])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#(4) Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#(5) Readout Layer
W_fc2 = weight_variable([64, label_size]) # 10 = length of output feature
b_fc2 = bias_variable([label_size]) #10 = length of output feature
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
                  
#(6) Train Variable
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
learning_rate=100.0*1e-5
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
prediction=y_conv; 
answer=y_;
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#(7) Session Start: Train
sess = tf.InteractiveSession()
sess.run( tf.initialize_all_variables())

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

for i_conv in range(50000):
    #Construct Training Data
    batch_x=[]; temp_x=[]; temp_y=[]; 
    for k in range(channels):
        batch_x.append(0);
        temp_x.append([]);
    #mini batch 
    for s in range(50):
         random_index=random.randint(0,int(xd1*0.9)-1)
         temp_y.append(list(training_label[random_index]))
         for k in range(channels):
            temp_x[k].append(np.array(training_data)[random_index,:,:,k])
    for k in range(channels):
        batch_x[k]=np.asarray(temp_x[k],dtype=float);
    batch_y=np.asarray(temp_y,dtype=float);

    #Print Training Accuracy for every 100th steps
    if i_conv%100 == 0:
        train_accuracy = sess.run(accuracy,feed_dict={x[0]:batch_x[0], x[1]:batch_x[1],  y_: batch_y, keep_prob: 1.0})
        print(" %dth Training Step, training accuracy %g"%(i_conv, train_accuracy))
        fo.write(" %dth Training Step, training accuracy %g\n"%(i_conv, train_accuracy))

    #Actual Training
    sess.run(train_step,feed_dict={x[0]:batch_x[0], x[1]:batch_x[1], y_: batch_y, keep_prob: 0.5})
    #(8)Print Acuuracy
    if i_conv%20 == 0:
        print("\t %d : test accuracy %g"%(i_conv,sess.run(accuracy,feed_dict={
             x[0]:test_data[:,:,:,0], x[1]:test_data[:,:,:,1], y_:test_label, keep_prob: 1.0})))
        fo.write("\t %d : test accuracy %g \n"%(i_conv,sess.run(accuracy,feed_dict={
             x[0]:test_data[:,:,:,0], x[1]:test_data[:,:,:,1],  y_:test_label, keep_prob: 1.0})))
pre=sess.run(prediction,feed_dict={x[0]:test_data[:,:,:,0], x[1]:test_data[:,:,:,1],  y_:test_label, keep_prob: 1.0})
ans=batch_y;
            
for k in range(len(ans)):
    print(k+1,np.argmax(ans[k]),np.argmax(pre[k]))
    fo.write(str(k+1)+", "+str(np.argmax(ans[k]))+" , "+str(np.argmax(pre[k])))

fo.close()
# Save the variables to disk.
save_path = saver.save(sess, "./detection_cnn.ckpt")
print("Model saved in file: %s" % save_path)
#
