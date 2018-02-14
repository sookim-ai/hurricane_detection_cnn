import tensorflow as tf
from tensorflow.contrib import rnn
from function import *
from inference import *
from load_data import *
from rnn import *
import numpy as np



def load_test_data(ii,path):
    te_image=np.load(path+"te_input_"+str(ii)+".npy")
    te_state=np.load(path+"te_output_state_"+str(ii)+".npy")
    te_lonlat=np.load(path+"te_output_lonlat_"+str(ii)+".npy")
    k1,k2,k3,k4,k5=np.shape(te_image);
    te_image=np.reshape(te_image[:,:,1:k3,:,:],[k1,k2,k3-1,k4*k5])
    te_state_out=te_state[:,:,1:k3,:]
    # Lonlat_in: Saved for masking mask
    te_lonlat_in=te_lonlat[:,:,0:k3-1,:]
    te_lonlat_out_out=te_lonlat[:,:,1:k3,:]
    return k1,k2,k3,k4,k5,te_image,te_state,te_lonlat,te_state_out,te_lonlat_in,te_lonlat_out_out



def test(name,sess,loss_op,train_op,X,Y_state_in,Y_lonlat_out,prediction_lonlat,last_state,iteration,path_test,path):
    zero_state=np.zeros((number_of_layers, 2, FLAGS.batch_size,lstm_size))
    for k in range(1):
        #Load data
        te_image=np.load(path+"va_input_batch.npy")
        te_state=np.load(path+"va_output_state_batch.npy")
        te_lonlat=np.load(path+"va_output_lonlat_batch.npy")
        fetches = {'final_state': last_state,
                   'prediction_lonlat': prediction_lonlat}
        lonlat_list=[]; state_list=[];
        t1,t2,t3,t4,t5=np.shape(te_image);
        te_image=np.reshape(te_image[:,:,1:t3,:,:],[t1,t2,t3-1,t4*t5])
        te_lonlat_in=te_lonlat[:,:,0:t3-1,:]
        te_lonlat_out=te_lonlat[:,:,1:t3,:]
        te_state_out=te_state[:,:,1:t3,:]
        fetches = {'final_state': last_state,
                   'prediction_lonlat': prediction_lonlat}
        lonlat_list=[]; state_list=[];

        ##THINGS TO DO:END_SIGNAL (Check) -  When feed Y_state_in, get output from pre-trained detection CNN
        ##First, let's load meta graph(detection CNN) and restore weights
        #saver_cnn=tf.train.import_meta_graph(path_to_checkpoint+'detection_cnn.ckpt.meta')
        #saver_cnn.restore(sess,tf.train.latest_checkpoint(path_to_checkpoint))
        ##Secondly, let's get graph(detection_cnn) and acess the specific tensor in graph by name 
        #detection_cnn = tf.get_default_graph()
        #state_prediction = detection_cnn.get_tensor_by_name("op_to_restore:0")
        #x0=detection_cnn.get_tensor_by_name("x0:0")
        #x1=detection_cnn.get_tensor_by_name("x1:0")
        #keep_p=detection_cnn.get_tensor_by_name("keep_p:0")

        for bch in range(t1):
            time=0
            initial_input_X = te_image[bch,:,time:time+1,:]  # put suitable data here size of dimension of input
            initial_input_Y_state_in = te_state[bch,:,time:time+1,:]
            initial_input_Y_lonlat_in = te_lonlat[bch,:,time:time+1,:]
            image=mask_around_lonlat(initial_input_X, initial_input_Y_lonlat_in)
            # get the output for the first time step
            feed_dict = {X:image, Y_state_in:initial_input_Y_state_in, init_state:zero_state}
            eval_out = sess.run(fetches, feed_dict)
            outputs_lonlat = [eval_out['prediction_lonlat']] #(1,24,1,2)
            next_state = eval_out['final_state']
            for time in xrange(1,t3-1):
                image= mask_around_lonlat(te_image[bch,:,time:time+1,:], outputs_lonlat[-1])

                ##THINGS TO DO: END_SIGNAL (Check)
                ##(1) Access to the 10x10cropped image([x0,x1]) from te_image[bch,:,time:time+1,:] centering around output_lonlat[-1]
                #cropped_image=crop_around_lonlat(image,y_lonlat_in) #(24,10,10,2) 
                #x0_val=cropped_image[:,:,:,0]; x1_val=cropped_image[:,:,:,1];
                ##(2) Obtain detection result from detection cnn
                #detection_results=sess.run(state_prediction,feed_dict={x0:x0_val,x1:x1_val,keep_p: 1.0}) #(24,2) yes->[0,1] no->[1,0]
                #detection_results= np.reshape(detection_results[:,1],[24,1,1]) #(24,1)-->(24,"1",1)
                #(3) Feed detection results as Y_state_in
                #feed_dict = {X:image ,Y_state_in:detection_results, Y_lonlat_in:y_lonlat_in, initial_state: next_state}

                feed_dict = {X:image ,Y_state_in:te_state[bch,:,time:time+1,:], init_state: next_state}
                eval_out = sess.run(fetches, feed_dict)
                outputs_lonlat.append(eval_out['prediction_lonlat'])
                next_state = eval_out['final_state']
            lonlat_list.append(np.asarray(outputs_lonlat)) #[(timesteps,24,batch(1),2),(),(), ...]
        print(np.shape(lonlat_list)); #(400, 3, 24, 1, 2)
        print(np.shape(te_lonlat_out)); #(400,24,3,2)
        print(np.shape(te_state_out));  #(400,24,3,1)
        te_lonlat=np.reshape(te_lonlat_out, [400,24,t3-1,1,2])
        lonlat_list=np.swapaxes(lonlat_list, 1,2)
        np.save("prediction_lonlat_"+str(it)+".npy",lonlat_list)
        np.save("ground_trunth_lonlat_"+str(it)+".npy",te_lonlat_out)
        np.save("ground_truth_state_",str(it)+".npy",te_state_out)

    for ii in range(12):
        #Load data
        k1,k2,k3,k4,k5,te_image,te_state,te_lonlat,te_state_out,te_lonlat_in,te_lonlat_out_out=load_test_data(ii,path_test)
        #Testing 
        fetches = {'final_state': last_state,
                   'prediction_lonlat': prediction_lonlat}
        lonlat_list=[]; state_list=[];
        for bch in range(k1):
            time=0
            initial_input_X = te_image[bch,:,time:time+1,:]  # put suitable data here size of dimension of input
            initial_input_Y_state_in = te_state[bch,:,time:time+1,:]
            y_lonlat_in = te_lonlat_in[bch,:,time:time+1,:]
            image=mask_around_lonlat(initial_input_X, y_lonlat_in)
            # get the output for the first time step
            feed_dict = {X:image, Y_state_in:initial_input_Y_state_in, init_state:zero_state}
            eval_out = sess.run(fetches, feed_dict)
            reconstructed_lonlat=eval_out['prediction_lonlat']
            outputs_lonlat=[reconstructed_lonlat]
            next_state = eval_out['final_state']
            for time in xrange(1,k3-1):
                image= mask_around_lonlat(te_image[bch,:,time:time+1,:], reconstructed_lonlat)
                initial_position=reconstructed_lonlat
                feed_dict = {X:image ,Y_state_in:te_state[bch,:,time:time+1,:], init_state: next_state}
                eval_out = sess.run(fetches, feed_dict)
                reconstructed_lonlat=eval_out['prediction_lonlat']
                print(eval_out['prediction_lonlat'][0,:,:],initial_position[0,:,:],reconstructed_lonlat[0,:,:])
                outputs_lonlat.append(reconstructed_lonlat)
                next_state = eval_out['final_state']
            lonlat_list.append(outputs_lonlat) #[(timesteps,24,batch(1),2),(),(), ...]
        te_lonlat_gt=te_lonlat[:,:,1:k3,:]
        te_lonlat_gt=np.reshape(te_lonlat_gt, [k1,k2,k3-1,1,2])
        lonlat_list=np.swapaxes(np.asarray(lonlat_list), 1,2) #(400,24,7,1,2)
        print(np.shape(lonlat_list))
        print(np.shape(te_lonlat_gt)); #(1,24,7,2)
        np.save("prediction_lonlat_"+name+"_"+str(ii)+".npy",lonlat_list)
        np.save("ground_trunth_lonlat_"+name+"_"+str(ii)+".npy",te_lonlat_gt)






