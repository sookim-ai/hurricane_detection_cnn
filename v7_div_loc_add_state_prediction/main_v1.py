from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow.contrib import rnn
from function import *
from inference import *
from rnn import *
import numpy as np
import skimage.measure
#State from ground truth, initial position:given, output location feed to input as mask
#Input: Masked Climate Image
#Output: (div_lon, div_lat) --> we can generate trajectories using ground truth initial position
#Things to do : implement "teacher forcing"

#1: Data Loading
fout_log= open("log.txt","w")
fout_log.write("TEST LOG PRINT\nSOO\n")
fout_pre= open("pred.txt","w")
fout_pres=open("pred_s.txt","w")
fout_gt=open("gt.txt","w")
fout_pre.write("\nPrediction \n"); fout_gt.write("\nGround Truth \n");
path_8="/export/kim79/h2/finished_npy/tracking_data/" #Hurricane can be off
path_to_checkpoint="/export/kim79/lstm_final/hurricane_detector_lr_2ch_good/more_data/"
#path_8="/export/kim79/h2/finished_npy/prediction_data/8/" #Hurricane is alwayas on


tr_input_image=np.load(path_8+"tr_input_batch.npy")
tr_output_state=np.load(path_8+"tr_output_state_batch.npy")
tr_output_lonlat=np.load(path_8+"tr_output_lonlat_batch.npy")
va_input_image=np.load(path_8+"te_input_batch.npy")
va_output_state=np.load(path_8+"te_output_state_batch.npy")
va_output_lonlat=np.load(path_8+"te_output_lonlat_batch.npy") 
te_image=np.load(path_8+"va_input_batch.npy")
te_state=np.load(path_8+"va_output_state_batch.npy")
te_lonlat=np.load(path_8+"va_output_lonlat_batch.npy")

d1,d2,d3,d4,d5=np.shape(va_input_image);
t1,t2,t3,t4,t5=np.shape(tr_input_image);
k1,k2,k3,k4,k5=np.shape(te_image);

timesteps=d3-1 #d3=8
train_size=t1;  val_size=d1;
feature_size=d4 #w*h
channels=d5
w=129;h=86;

#Input tstep:1 to tstep 8
#(batch_number, batch_size, timesteps, 2)
tr_image=np.reshape(tr_input_image[:,:,1:d3,:,:],[t1,t2,t3-1,t4*t5]) #(1500,24,d3-1, -1)
va_image=np.reshape(va_input_image[:,:,1:d3,:,:],[d1,d2,d3-1,d4*d5])
te_image=np.reshape(te_image[:,:,1:k3,:,:],[k1,k2,k3-1,k4*k5])

tr_state_in=tr_output_state[:,:,1:d3,:]
va_state_in=va_output_state[:,:,1:d3,:]
te_state_out=te_state[:,:,1:d3,:]

# Lonlat_in: Saved for generating mask
tr_lonlat_in=tr_output_lonlat[:,:,0:d3-1,:] 
va_lonlat_in=va_output_lonlat[:,:,0:d3-1,:]
te_lonlat_in=te_lonlat[:,:,0:d3-1,:]

#Change as div_of_locations:1/0 to 8/7 
tr_lonlat_out_div=div_of_lonlat(tr_output_lonlat[:,:,0:d3,:]) # div(1,0) to div(d3, d3-1)
va_lonlat_out_div=div_of_lonlat(va_output_lonlat[:,:,0:d3,:]) 
te_lonlat_out_div=div_of_lonlat(te_lonlat[:,:,0:d3,:])

#2: Graph
#Training Parameters
validation_step=10;
learning_rate =0.0005
training_steps =200000 
X = tf.placeholder("float", [FLAGS.batch_size, None, feature_size*channels])
Y_state_in = tf.placeholder("float", [FLAGS.batch_size, None, 1])
Y_lonlat_out = tf.placeholder("float", [FLAGS.batch_size, None, 2])
timesteps = tf.shape(X)[1]

xx=Inference(X,timesteps)
prediction_lonlat, last_state = RNN(xx, weights, biases)

state_sum=tf.reduce_sum(Y_state_in) #total number of 1 in Y_state_in
state=tf.concat([Y_state_in,Y_state_in],axis=2) 
prediction_lonlat=tf.multiply(prediction_lonlat,state) # prediction value when state prediction is 1
sqsum=tf.reduce_sum(tf.pow(prediction_lonlat - Y_lonlat_out,2))
loss_lonlat=tf.div(sqsum,state_sum) #Ave_MSE pre one element
loss_op=loss_lonlat*16000 #unit of km
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

with tf.Session() as sess:
    #3: Training
    # Initialize all variables
    #saver = tf.train.Saver()
    val_best_loss=0.001;
    val_best_step=0;
    num_epoch=0;
    iteration=100;
    alpha=0.9
    seq_length=7
    init = tf.global_variables_initializer()
    sess.run(init)
    fetches = {'final_state': last_state,
              'prediction_lonlat': prediction_lonlat}
    for it in range(iteration):
    ###Training Phase 1 ####################    
        zero_state=np.zeros((number_of_layers, 2, FLAGS.batch_size,lstm_size))
        for step in range(0,2*num_epoch*train_size):
            step_i=np.random.randint(0,train_size);
            j=np.random.randint(0,val_size);
            #Generate mask
            image=mask_around_lonlat(tr_image[step_i],tr_lonlat_in[step_i])
            train=sess.run(train_op, feed_dict={X:image, Y_state_in:tr_state_in[step_i], Y_lonlat_out:tr_lonlat_out_div[step_i], init_state: zero_state})
            # Calculate batch loss in validation set
            image_va=mask_around_lonlat(va_image[j],va_lonlat_in[j])
            loss= sess.run(loss_op,  feed_dict={X:image_va, Y_state_in:va_state_in[j], Y_lonlat_out:va_lonlat_out_div[j], init_state: zero_state})
            #Calculated running average of val_loss
            if step == 10:  # loss start from very large val at step=0, so start from 10th step
                val_loss = loss
            elif step >10:
                val_loss = alpha * val_loss + (1-alpha) * loss
                #write up
                fout_log.write("Step " + str(step) + ", Validation Loss= " + \
                          "{:.4f}".format(val_loss) + ", Loss= "+\
                          "{:.4f}".format(loss) + "\n")
                print("Iteration "+str(it)+" Phase 1 Step " + str(step) + ", Validation Loss= " + \
                          "{:.4f}".format(val_loss) + ", Loss= "+\
                          "{:.4f}".format(loss) + "\n")
            if step > 10 and val_loss < val_best_loss:
                val_best_loss = val_loss
                print('found new best validation loss:', val_loss)
        print("Iteration "+str(it)+"Training DONE! \n Start Training Phase 2 -Train with Prediction \n")

        ###Training Phase 2 : Train with prediction ####################
        for step in range(0,1): # num_epoch*train_size):
            step_i=np.random.randint(0,train_size);
            j=np.random.randint(0,val_size);
            time=0
            image=mask_around_lonlat(tr_image[step_i,:,time:time+1,:],tr_lonlat_in[step_i,:,time:time+1,:])
            feed_dict={X:image, Y_state_in:tr_state_in[step_i,:,time:time+1,:], Y_lonlat_out:tr_lonlat_out_div[step_i,:,time:time+1,:], init_state:zero_state}
            sess.run(train_op, feed_dict)
            eval_out=sess.run(fetches,feed_dict)
            next_state=eval_out['final_state']
            y_lonlat_in_div=eval_out['prediction_lonlat']
            y_lonlat_in=reconstruct_one_lonlat(y_lonlat_in_div,tr_lonlat_in[step_i,:,time:time+1,:])
            for time in xrange(1,d3-1):
                image=mask_around_lonlat(tr_image[step_i,:,time:time+1,:],y_lonlat_in)
                feed_dict={X:image, Y_state_in:tr_state_in[step_i,:,time:time+1,:], Y_lonlat_out:tr_lonlat_out_div[step_i,:,time:time+1,:],init_state:next_state}
                train=sess.run(train_op, feed_dict)
                eval_out=sess.run(fetches,feed_dict)
                next_state=eval_out['final_state']
                y_lonlat_in_div=eval_out['prediction_lonlat']
                y_lonlat_in=reconstruct_one_lonlat(y_lonlat_in_div,y_lonlat_in)
            # Calculate batch loss in validation set
            time=0
            image_va=mask_around_lonlat(va_image[j,:,time:time+1,:],va_lonlat_in[j,:,time:time+1,:])
            feed_dict={X:image_va, Y_state_in:va_state_in[j,:,time:time+1,:], Y_lonlat_out:va_lonlat_out_div[j,:,time:time+1,:],init_state:zero_state}
            eval_out=sess.run(fetches,feed_dict)
            next_state=eval_out['final_state']
            y_lonlat_in_div=eval_out['prediction_lonlat']
            y_lonlat_in=reconstruct_one_lonlat(y_lonlat_in_div,va_lonlat_in[j,:,time:time+1,:])
            for time in xrange(1,d3-1):
                image=mask_around_lonlat(va_image[j,:,time:time+1,:],y_lonlat_in)
                feed_dict={X:image, Y_state_in:va_state_in[j,:,time:time+1,:], Y_lonlat_out:va_lonlat_out_div[j,:,time:time+1,:],init_state:next_state}
                loss= sess.run(loss_op, feed_dict)
                eval_out=sess.run(fetches,feed_dict)
                next_state=eval_out['final_state']
                y_lonlat_in_div=eval_out['prediction_lonlat']
                y_lonlat_in=reconstruct_one_lonlat(y_lonlat_in_div,y_lonlat_in)
                #Calculated running average of val_loss
                if step == 10:  # loss start from very large val at step=0, so start from 10th step
                    val_loss = loss
                elif step >10:
                    val_loss = alpha * val_loss + (1-alpha) * loss
                    #write up
                    fout_log.write("Step " + str(step) + ", Validation Loss= " + \
                              "{:.4f}".format(val_loss) + ", Loss= "+\
                              "{:.4f}".format(loss) + "\n")
                    if step%10 == 0:
                               print("Iteration "+str(it)+ "Training Phase 2 Step " + str(step) + ", Validation Loss= " + \
                                   "{:.4f}".format(val_loss) + ", Loss= "+\
                                   "{:.4f}".format(loss) + "\n")
                if step > 10 and val_loss < val_best_loss:
                    val_best_loss = val_loss
                    #saver.save(sess, os.path.join(train_dir, 'model-val_best'), global_step=step)
                    print('found new best validation loss:', val_loss)
        print("Iteration "+str(it)+" Training Phase 2 -  Train with prediction DONE! \n Start Testing \n")



        #3: Testing  
        fetches = {'final_state': last_state,
                   'prediction_lonlat': prediction_lonlat}
        lonlat_list=[]; state_list=[];

        #THINGS TO DO:END_SIGNAL (Check) -  When feed Y_state_in, get output from pre-trained detection CNN
        #First, let's load meta graph(detection CNN) and restore weights
        saver_cnn=tf.train.import_meta_graph(path_to_checkpoint+'detection_cnn.ckpt.meta')
        saver_cnn.restore(sess,tf.train.latest_checkpoint(path_to_checkpoint))
        #Secondly, let's get graph(detection_cnn) and acess the specific tensor in graph by name 
        detection_cnn = tf.get_default_graph()
        state_prediction = detection_cnn.get_tensor_by_name("op_to_restore:0")
        x0=detection_cnn.get_tensor_by_name("x0:0")
        x1=detection_cnn.get_tensor_by_name("x1:0")
        keep_p=detection_cnn.get_tensor_by_name("keep_p:0")
        prediction_state_batch=[];

        for bch in range(k1):
            time=0
            initial_input_X = te_image[bch,:,time:time+1,:]  # put suitable data here size of dimension of input
            initial_input_Y_state_in = te_state[bch,:,time:time+1,:] 
            y_lonlat_in = te_lonlat_in[bch,:,time:time+1,:]  
            image=mask_around_lonlat(initial_input_X, y_lonlat_in)
            # get the output for the first time step
            feed_dict = {X:image, Y_state_in:initial_input_Y_state_in, init_state:zero_state}
            eval_out = sess.run(fetches, feed_dict)
            outputs_lonlat = [eval_out['prediction_lonlat']] #(1,24,1,2)
            next_state = eval_out['final_state']
            prediction_state_time=[initial_input_Y_state_in];
            for time in xrange(1,k3-1):
                y_lonlat_in=reconstruct_one_lonlat(outputs_lonlat[-1],y_lonlat_in)
                image= mask_around_lonlat(te_image[bch,:,time:time+1,:], y_lonlat_in)

                #THINGS TO DO: END_SIGNAL (Check)
                #(1) Access to the 10x10cropped image([x0,x1]) from te_image[bch,:,time:time+1,:] centering around output_lonlat[-1]
                cropped_image=crop_around_lonlat(image,y_lonlat_in) #(24,10,10,2) 
                x0_val=cropped_image[:,:,:,0]; x1_val=cropped_image[:,:,:,1];
                #(2) Obtain detection result from detection cnn
                detection_results=sess.run(state_prediction,feed_dict={x0:x0_val,x1:x1_val,keep_p: 1.0}) #(24,2) yes->[0,1] no->[1,0]
                detection_results= np.reshape(detection_results[:,1],[24,1,1]) #(24,1)-->(24,"1",1)
                #(3) Feed detection results as Y_state_in
                feed_dict = {X:image ,Y_state_in:detection_results, init_state: next_state}

                #feed_dict = {X:image ,Y_state_in:te_state[bch,:,time:time+1,:], init_state: next_state}
                eval_out = sess.run(fetches, feed_dict)
                outputs_lonlat.append(eval_out['prediction_lonlat'])
                next_state = eval_out['final_state']
            lonlat_list.append(outputs_lonlat) #[(timesteps,24,batch(1),2),(),(), ...]
            prediction_state_time=np.concatenate(prediction_state_time,1)
            prediction_state_batch.append([prediction_state_time])
        prediction_state_batch=np.asarray(np.concatenate(prediction_state_batch,0))
        te_lonlat_gt=te_lonlat[:,:,2:d3,:]
        print(np.shape(prediction_state_batch))
        print(np.shape(lonlat_list)); #(400, 6, 24, 1, 2)
        print(np.shape(te_lonlat_gt)); #(400,24,6,2)
        print(np.shape(te_state_out));  #(400,24,6,1)
        te_lonlat_gt=np.reshape(te_lonlat_gt, [k1,k2,k3-1,1,2])
        lonlat_list=np.swapaxes(np.asarray(lonlat_list), 1,2) #(400,24,6,1,2)
        print(np.shape(lonlat_list))
        #Reconstrunction
        te_init=te_lonlat[:,:,1:2,:]
        lonlat_list=reconstruct_div_to_lonlat(lonlat_list,te_init)
        np.save("prediction_lonlat_"+str(it)+".npy",lonlat_list)
        np.save("ground_trunth_lonlat_"+str(it)+".npy",te_lonlat_gt)
        np.save("ground_truth_state_",str(it)+".npy",te_state_out)
        np.save("prediction_state_",str(it)+".npy",prediction_state_batch)


########################################################################
fout_log.close();
fout_pre.close();
fout_gt.close();


