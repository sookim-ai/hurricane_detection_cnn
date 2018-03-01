import tensorflow as tf
from function import *
import numpy as np
import skimage.measure

def load_data(path):
    tr_input_image=np.load(path+"tr_input_batch.npy")
    tr_output_state=np.load(path+"tr_output_state_batch.npy")
    tr_output_lonlat=np.load(path+"tr_output_lonlat_batch.npy")
    va_input_image=np.load(path+"va_input_batch.npy")
    va_output_state=np.load(path+"va_output_state_batch.npy")
    va_output_lonlat=np.load(path+"va_output_lonlat_batch.npy")

    d1,d2,d3,d4,d5=np.shape(va_input_image);
    t1,t2,t3,t4,t5=np.shape(tr_input_image);

    timesteps=d3-1 #d3=8
    train_size=t1;  val_size=d1;
    channels=d5

    #Input tstep:2 to tstep 8
    #(batch_number, batch_size, timesteps, 2)
    tr_image=np.reshape(tr_input_image[:,:,1:d3,:,:],[t1,t2,t3-1,t4*t5]) #(1500,24,d3-1, -1)
    va_image=np.reshape(va_input_image[:,:,1:d3,:,:],[d1,d2,d3-1,d4*d5])

    tr_state_in=tr_output_state[:,:,1:d3,:]
    va_state_in=va_output_state[:,:,1:d3,:]

    # Lonlat_in: Saved for masking mask
    tr_lonlat_in=tr_output_lonlat[:,:,0:d3-1,:]
    va_lonlat_in=va_output_lonlat[:,:,0:d3-1,:]

    #Change as div_of_locations:1/0 to 8/7 lonlat_out: 2/1 to 8/7
    tr_lonlat_out_div=div_of_lonlat(tr_output_lonlat[:,:,0:d3,:]) # div(1,0) to div(d3, d3-1)
    va_lonlat_out_div=div_of_lonlat(va_output_lonlat[:,:,0:d3,:])
    return d1,d2,d3,d4,d5,t1,t2,t3,t4,t5,timesteps,train_size,val_size,channels,tr_image,va_image,tr_state_in,va_state_in,tr_lonlat_in,va_lonlat_in,tr_lonlat_out_div,va_lonlat_out_div

