from random import shuffle
import numpy as np
import copy
import skimage.measure
fname="/export/kim79/h2/sort_by_time.csv"
gname="../sort_by_track_all.csv"
# We have dataset with traject number #233 
ts=8;

#(15000, 128, 288, 1)
h=128
w=288;
ch=2;
#READ input
inn=np.load("/export/kim79/h2/finished_npy/nopool_0208_heatmap_0208.npy"); #40~80
###########CROP###########
inn=inn[:,:,:,:]
input_data=skimage.measure.block_reduce(inn, (1,2,2,1), np.max);
input_data=skimage.measure.block_reduce(input_data, (1,2,2,1), np.max); #(47, 128,288, 3)
print(np.shape(input_data))
#1/4x1/4 upscalig

sh1,sh2,sh3,sh4=np.shape(input_data);
input_data= list(np.reshape(input_data,[sh1,sh2*sh3,sh4]))

#READ original input sort_by_time.csv which are the sequence of inn
with open(fname) as f:
    content = f.readlines();
    time_list=[];lat_list=[]; lon_list=[]; num_list=[]; traj_list=[];
    time_list.append(content[0].split(",")[0])
    for i in range(len(content)):
        #input
        line=content[i].split(",")
        time=line[0];lon=line[1];lat=line[2];track_id=line[3]; tc_id=line[4];
        time_list.append(time); lon_list.append(lon); lat_list.append(lat); traj_list.append(track_id); num_list.append(tc_id);

#READ Target input sort_by_track.csv
with open(gname) as g:
    content = g.readlines();
    time_list2=[];lat_list2=[]; lon_list2=[]; num_list2=[]; traj_list2=[];
    time_list2.append(content[0].split(",")[0])
    for i in range(9460): ##REFER sort_by_track3.csv
        #input
        line=content[i].split(",")
        time=line[0];lon=line[1];lat=line[2];track_id=line[3]; tc_id=line[4];
        time_list2.append(time); lon_list2.append(lon); lat_list2.append(lat); traj_list2.append(track_id); num_list2.append(tc_id);

#Re-order input file
input_data2=[];
for i in range(9460):
    element =input_data[num_list.index(num_list2[i])]
    input_data2.append(element);
print("CHECK")


a='0'
listt=[];
for i in range(len(traj_list2)):
    if(traj_list2[i]!=a):
        listt.append(traj_list2[i]);
        a=traj_list2[i]

#Making Data
batch_size=24;
ts=timestep=8;
def read_input(train_size):
    input_image=[]; output_state=[]; output_lonlat=[];
    for i in range(len(listt)-1):
        temp_state=[]; temp_lonlat=[];
        start_index=traj_list2.index(listt[i]);
        if i==len(listt)-1: 
            end_index=20000;
        else : 
            end_index=traj_list2.index(listt[i+1])-1;
        if (end_index - start_index) >timestep:
            
            for k in range(int(((end_index-start_index)-timestep)/(ts/2))):
                input_image.append(input_data2[start_index+k*(ts/2):start_index+timestep+k*(ts/2)])
                temp_state=[]; temp_lonlat=[];
                for j in xrange(start_index+k*(ts/2),start_index+k*(ts/2)+timestep):
                    temp_state.append([1]);
                    temp_lonlat.append([(float(lon_list2[j]))/360.0,((float(lat_list2[j])+40.0)/120.0)])
                output_state.append(temp_state);
                output_lonlat.append(temp_lonlat);
    #input_image=np.reshape(input_image,[-1,batch_size, 11094,2])
    print(np.shape(input_image),np.shape(output_state),np.shape(output_lonlat));
    ###Make as batch##############
    input_batch=[]; output_state_batch=[]; output_lonlat_batch=[];
    for i in range(end_index-24):    
        input_batch.append(input_image[i:i+batch_size]);
        output_state_batch.append(output_state[i:i+batch_size]);
        output_lonlat_batch.append(output_lonlat[i:i+batch_size]);
    #shuffle(index)
    tr_input_batch=[]; tr_output_state_batch=[]; tr_output_lonlat_batch=[];
    va_input_batch=[]; va_output_state_batch=[]; va_output_lonlat_batch=[];
    te_input_batch=[]; te_output_state_batch=[]; te_output_lonlat_batch=[];
    for i in range(train_size):
        tr_input_batch.append(input_batch[i]);
        tr_output_state_batch.append(output_state_batch[i]);
        tr_output_lonlat_batch.append(output_lonlat_batch[i]);
    for j in xrange(train_size,train_size+50):
        va_input_batch.append(input_batch[j]);
        va_output_state_batch.append(output_state_batch[j]);
        va_output_lonlat_batch.append(output_lonlat_batch[j]);
    print(np.shape(tr_input_batch), np.shape(va_input_batch))
    print(np.shape(tr_output_state_batch), np.shape(va_output_state_batch))
    print(np.shape(tr_output_lonlat_batch),np.shape(va_output_lonlat_batch))
    np.save("./tr_input_batch.npy",tr_input_batch); 
    np.save("./tr_output_state_batch.npy",np.asarray(tr_output_state_batch))
    np.save("./tr_output_lonlat_batch.npy",np.asarray(tr_output_lonlat_batch))
    np.save("./va_input_batch.npy",np.asarray(va_input_batch))
    np.save("./va_output_state_batch.npy",va_output_state_batch)
    np.save("./va_output_lonlat_batch.npy",np.asarray(va_output_lonlat_batch))

read_input(1850);


