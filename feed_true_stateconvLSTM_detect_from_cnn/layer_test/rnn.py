from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow.contrib import rnn
from inference import *

w=129; h=86;
display_step=10;
testing_step=100;
training_steps = 200000
timesteps=7;
feature_size=w*h;
# Network Parameters
lstm_size=num_hidden = 200 # hidden layer num of features
number_of_layers=4; #Start from only one layer


def lstm_cell():
  return tf.contrib.rnn.BasicLSTMCell(lstm_size)
stacked_lstm=tf.contrib.rnn.MultiRNNCell(
    [lstm_cell() for _ in range(number_of_layers)])

initial_state = stacked_lstm.zero_state(FLAGS.batch_size, tf.float32)

# Define weights localization output
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, 2]))
}
biases = {
    'out': tf.Variable(tf.random_normal([2]))
}


def RNN(x, weights, biases):
    # x:shape=(24, 7, 4096):[batch_size, seq_lenth,dimension of input
    outputs, last_states = tf.nn.dynamic_rnn(cell=stacked_lstm, inputs=x , sequence_length=None, dtype=tf.float32, initial_state=initial_state); #output:(24,7,170):[batch_size, seq_length,hidden_states] states:(24,7):[batch_size, seq_length]
    t1,t2,t3=outputs.get_shape().as_list()
    outputs = tf.reshape(outputs, [-1,t3])
    out_lonlat=tf.matmul(outputs, weights['out']) + biases['out'];
    output_lonlat=tf.reshape(out_lonlat, [t1, -1, 2])

    return output_lonlat ,last_states




