import tensorflow as tf
from tensorflow.contrib import rnn
from function import *
from inference import *
from load_data import *
from rnn import *
import numpy as np

def train(sess,loss_op,train_op,X,Y_state_in,Y_lonlat_out,prediction_lonlat,last_state,iteration,path,fout_log):
    val_best_loss=0.05;
    val_best_step=0;
    num_epoch=2;
    alpha=0.9
    beta= 0.95 # teacher's forcing
    percent_of_ground_truth=1
    fetches = {'final_state': last_state,
              'prediction_lonlat': prediction_lonlat}
    d1,d2,d3,d4,d5,t1,t2,t3,t4,t5,timesteps,train_size,val_size,channels,tr_image,va_image,tr_state_in,va_state_in,tr_lonlat_in,va_lonlat_in,tr_lonlat_out_div,va_lonlat_out_div=load_data(path)
    for it in range(iteration):
        percent_of_ground_truth= beta * percent_of_ground_truth
        percent_of_prediction= 1-percent_of_ground_truth

        ### Training Phase 1 ####################
        zero_state=np.zeros((number_of_layers, 2, FLAGS.batch_size,lstm_size))
        for step in range(0,int(percent_of_ground_truth*num_epoch*train_size)):
            step_i=np.random.randint(0,train_size);
            j=np.random.randint(0,val_size);
            #Generate mask
            image=mask_around_lonlat(tr_image[step_i],tr_lonlat_in[step_i])
            train=sess.run(train_op, feed_dict={X:image, Y_state_in:tr_state_in[step_i], Y_lonlat_out:tr_lonlat_out_div[step_i], init_state: zero_state})
            # Calculate batch loss in validation set
            if step>10 and step%100 == 0:
                val_sum=0
                for j in range(val_size):
                    image_va=mask_around_lonlat(va_image[j],va_lonlat_in[j])
                    lossv= sess.run(loss_op,  feed_dict={X:image_va, Y_state_in:va_state_in[j], Y_lonlat_out:va_lonlat_out_div[j], init_state: zero_state})
                    val_sum=val_sum+lossv
                loss = val_sum/(val_size)
                #Calculated running average of val_loss
                if step == 100:  # loss start from very large val at step=0, so start from 10th step
                    val_loss = loss
                elif step >100:
                    val_loss = alpha * val_loss + (1-alpha) * loss
                    #write up
                    fout_log.write("Step " + str(step) + ", Validation Loss= " + \
                          "{:.4f}".format(val_loss ** 0.5) + ", Loss= "+\
                          "{:.4f}".format(loss) + "\n")
                    print("Iteration "+str(it)+" Phase 1 Step " + str(step) + ", Validation Loss= " + \
                              "{:.4f}".format(val_loss ** 0.5) + ", Loss= "+\
                              "{:.4f}".format(loss) + "\n")
                if step > 100 and val_loss < val_best_loss:
                    val_best_loss = val_loss
                    save_path = saver.save(sess, "./model.ckpt")
                    print('found new best validation loss:', val_loss)
                    print("Model saved in path: %s" % save_path)
                    count = 0
                if ((it)*int(percent_of_ground_truth*num_epoch*train_size)+step > 2*train_size) and (val_loss > val_best_loss):
                    count =  count + 1
                    if (count > 5):
                        print("Iteration "+str(it)+"Training DONE! \n Start Training Phase 2 -Train with Prediction \n")
                        break


        ###Training Phase 2 : Train with prediction(Teacher_forcing)  ####################
        for step in range(0,int(percent_of_prediction*num_epoch*train_size)):
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
            for time in xrange(1,timesteps):
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
        print("Iteration "+str(it)+" Training Phase 2 -  Train with prediction DONE! \n")



#def test(sess,X,Y_state_in,Y_lonlat_out,prediction_lonlat,last_state,iteration,path):




