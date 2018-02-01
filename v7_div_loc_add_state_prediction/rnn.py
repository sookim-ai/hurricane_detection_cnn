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
  return tf.nn.rnn_cell.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(lstm_size),output_keep_prob=0.5)
stacked_lstm=tf.contrib.rnn.MultiRNNCell(
    [lstm_cell() for _ in range(number_of_layers)])


init_state=tf.placeholder(tf.float32, [number_of_layers, 2, FLAGS.batch_size,lstm_size])
state_per_layer_list = tf.unstack(init_state, axis=0)
rnn_tuple_state = tuple(
    [tf.nn.rnn_cell.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
     for idx in range(number_of_layers)]
)



# Define weights localization output
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, 2]))
}
biases = {
    'out': tf.Variable(tf.random_normal([2]))
}


def RNN(x, weights, biases):
    outputs, last_states = tf.nn.dynamic_rnn(cell=stacked_lstm, inputs=x , sequence_length=None, dtype=tf.float32, initial_state=rnn_tuple_state)
    t1,t2,t3=outputs.get_shape().as_list()
    outputs = tf.reshape(outputs, [-1,t3])
    out_lonlat=tf.matmul(outputs, weights['out']) + biases['out'];
    output_lonlat=tf.reshape(out_lonlat, [t1, -1, 2])
    last_states=tf.stack(last_states,axis=0)
    return output_lonlat ,last_states




