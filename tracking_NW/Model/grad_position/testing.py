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
    #Change as div_of_locations:1/0 to 8/7 lonlat_out: 2/1 to 8/7
    te_lonlat_out_div=div_of_lonlat(te_lonlat[:,:,0:k3,:])
    return k1,k2,k3,k4,k5,te_image,te_state,te_lonlat,te_state_out,te_lonlat_in,te_lonlat_out_div









def test(name,sess,loss_op,train_op,X,Y_state_in,Y_lonlat_out,prediction_lonlat,last_state,iteration,path_test):
    zero_state=np.zeros((number_of_layers, 2, FLAGS.batch_size,lstm_size))
    for ii in range(12):
        #Load data
        k1,k2,k3,k4,k5,te_image,te_state,te_lonlat,te_state_out,te_lonlat_in,te_lonlat_out_div=load_test_data(ii,path_test)
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
            reconstructed_lonlat=reconstruct_one_lonlat(eval_out['prediction_lonlat'],y_lonlat_in)
            outputs_lonlat=[reconstructed_lonlat]
            next_state = eval_out['final_state']
            for time in xrange(1,k3-1):
                image= mask_around_lonlat(te_image[bch,:,time:time+1,:], reconstructed_lonlat)
                initial_position=reconstructed_lonlat
                feed_dict = {X:image ,Y_state_in:te_state[bch,:,time:time+1,:], init_state: next_state}
                eval_out = sess.run(fetches, feed_dict)
                reconstructed_lonlat=reconstruct_one_lonlat(eval_out['prediction_lonlat'],initial_position)
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






