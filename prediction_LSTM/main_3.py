from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow.contrib import rnn
from inference import *
from rnn import *
import numpy as np
import skimage.measure

#1: Training
fout_log= open("log.txt","w")
fout_log.write("TEST LOG PRINT\nSOO\n")
fout_pre= open("pred.txt","w")
fout_pres=open("pred_s.txt","w")
fout_gt=open("gt.txt","w")
fout_pre.write("\nPrediction \n"); fout_gt.write("\nGround Truth \n");
# Data Loading
path="/export/kim79/h2/finished_npy/tracking_data/" # slow and long data: 8
#path="/export/kim79/h2/finished_npy/7/" #fast and short data: 4
tr_input_image=np.load(path+"tr_input_batch.npy")
tr_output_state=np.load(path+"tr_output_state_batch.npy")
tr_output_lonlat=np.load(path+"tr_output_lonlat_batch.npy")
va_input_image=np.load(path+"va_input_batch.npy")
va_output_state=np.load(path+"va_output_state_batch.npy")
va_output_lonlat=np.load(path+"va_output_lonlat_batch.npy")
# Data Modulation
d1,d2,d3,d4,d5=np.shape(va_input_image);
t1,t2,t3,t4,t5=np.shape(tr_input_image);

timesteps=d3-1
train_size=t1;  val_size=d1;
feature_size=d4 #w*h
channels=d5
w=129;h=86;
tr_image=np.reshape(tr_input_image[:,:,1:d3,:,:],[t1,t2,timesteps,t4*t5])
va_image=np.reshape(va_input_image[:,:,1:d3,:,:],[d1,d2,timesteps,d4*d5])
#Trouble Shooting
tr_lonlat_in=np.concatenate((tr_output_lonlat[:,:,1:2,:],tr_output_lonlat[:,:,1:d3-1,:]), axis=2) #Given initial position
va_lonlat_in=np.concatenate((va_output_lonlat[:,:,1:2,:],va_output_lonlat[:,:,1:d3-1,:]), axis=2)
tr_state_in=tr_output_state[:,:,1:d3,:]
va_state_in=va_output_state[:,:,1:d3,:]
tr_lonlat_out=tr_output_lonlat[:,:,1:d3,:]
va_lonlat_out=va_output_lonlat[:,:,1:d3,:]

#Training Parameters
validation_step=10;
learning_rate =0.0005
training_steps =200000 
#TF Graph input
#CORRECTED timesteps to None
X = tf.placeholder("float", [FLAGS.batch_size, None, feature_size*channels])
Y_state_in = tf.placeholder("float", [FLAGS.batch_size, None, 1])
Y_lonlat_in = tf.placeholder("float", [FLAGS.batch_size, None, 2])
Y_lonlat_out = tf.placeholder("float", [FLAGS.batch_size, None, 2])

#Main function
x=tf.concat([X,Y_state_in,Y_lonlat_in], 2) 
prediction_lonlat, last_state = RNN(x, weights, biases)

#LOSS: Things to do
state_sum=tf.reduce_sum( tf.reduce_sum( tf.reduce_sum(Y_state_in,2),1),0) #total number of 1 in Y_state_in
state=tf.concat([Y_state_in,Y_state_in],axis=2) 
prediction_lonlat=tf.multiply(prediction_lonlat,state) # prediction value when state prediction is 1
# Define loss and optimizer
sqsum=tf.reduce_sum(tf.pow(prediction_lonlat - Y_lonlat_out,2))
loss_lonlat=tf.div(sqsum,state_sum) #Ave_MSE pre one element
loss_op=loss_lonlat 
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

with tf.Session() as sess:
    # Initialize all variables
    saver = tf.train.Saver()
    val_best_loss=0.001;
    val_best_step=0;
    num_epoch=10;
    alpha=0.9
    seq_length=7
    train_dir = 'best_model'
    os.system('mkdir -p {}'.format(train_dir))
    init = tf.global_variables_initializer()
    sess.run(init)
    for step in range(0, 100): #num_epoch*train_size):
        #Load Data
        step_i=int(step%train_size);
        j=int(step%val_size);
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X:tr_image[step_i], Y_state_in:tr_state_in[step_i], Y_lonlat_in:tr_lonlat_in[step_i], Y_lonlat_out:tr_lonlat_out[j]})
        # Calculate batch loss in validation set
        loss= sess.run(loss_op, feed_dict={X:tr_image[j], Y_state_in:tr_state_in[j], Y_lonlat_in:tr_lonlat_in[j], Y_lonlat_out:tr_lonlat_out[j]})
        #Calculated running average of val_loss
        if step == 10:  # loss start from very large val at step=0, so start from 10th step
            val_loss = loss
        elif step >10:
            val_loss = alpha * val_loss + (1-alpha) * loss
            #write up
            fout_log.write("Step " + str(step) + ", Validation Loss= " + \
                      "{:.4f}".format(val_loss) + ", Loss= "+\
                      "{:.4f}".format(loss) + "\n")
            print("Step " + str(step) + ", Validation Loss= " + \
                      "{:.4f}".format(val_loss) + ", Loss= "+\
                      "{:.4f}".format(loss) + "\n")
        if step > 10 and val_loss < val_best_loss:
            val_best_loss = val_loss
            saver.save(sess, os.path.join(train_dir, 'model-val_best'), global_step=step)
            print('found new best validation loss:', val_loss)
    print("DONE and writing test data")
    #2: Testing te_state_in/out and te_lonlat_in/out should be feed in from previous t-step
    te_image=np.load(path+"te_input_batch.npy")
    te_state=np.load(path+"te_output_state_batch.npy")
    te_lonlat=np.load(path+"te_output_lonlat_batch.npy")
    sh1,sh2,sh3,sh4,sh5=np.shape(te_image)
    te_image=np.asarray(np.reshape(te_image,[sh1,sh2,sh3,sh4*sh5]))
    fetches = {'final_state': last_state,
               'prediction_lonlat': prediction_lonlat}
    lonlat_list=[]; state_list=[]; 
    for bch in range(sh1):
        time=0
        initial_input_X = te_image[bch,:,time:time+1,:]  # put suitable data here size of dimension of input
        initial_input_Y_state_in = te_state[bch,:,time:time+1,:] # np.zeros([FLAGS.batch_size, 1, 1]) #Think whether its correct
        initial_input_Y_lonlat_in = te_lonlat[bch,:,time:time+1,:]  #np.zeros([FLAGS.batch_size, 1, 2])

        # get the output for the first time step
        feed_dict = {X:initial_input_X, Y_state_in:initial_input_Y_state_in, Y_lonlat_in:initial_input_Y_lonlat_in}
        eval_out = sess.run(fetches, feed_dict)
        outputs_lonlat = [eval_out['prediction_lonlat']] #(1,24,1,2)
        next_state = eval_out['final_state']
        for time in range(1,sh3):
            print(np.shape(te_image[bch,:,time:time+1,:]))
            print(np.shape(outputs_lonlat[-1]))
            np.save("image.npy",te_image[bch,:,time:time+1,:])
            np.save("y_lonlat_in.npy", outputs_lonlat[-1])
            exit(7);
            feed_dict = {X:te_image[bch,:,time:time+1,:] ,Y_state_in:te_state[bch,:,time:time+1,:], Y_lonlat_in: outputs_lonlat[-1], initial_state: next_state}
            eval_out = sess.run(fetches, feed_dict)
            outputs_lonlat.append(eval_out['prediction_lonlat'])
            next_state = eval_out['final_state']
        lonlat_list.append(np.asarray(outputs_lonlat)) #(timestes,24,batch(1),2)
    outputs_lonlat=np.concatenate(lonlat_list,axis=2)
    outputs_lonlat=np.swapaxes(outputs_lonlat,0,2)
    print(np.shape(outputs_lonlat)); 
    print(np.shape(te_lonlat));
    print(np.shape(te_state)); 
    np.save("prediction_lonlat.npy",lonlat_list)
    np.save("ground_trunth_lonlat.npy",te_lonlat)
    np.save("ground_truth_state.npy",te_state)

########################################################################
fout_log.close();
fout_pre.close();
fout_gt.close();


