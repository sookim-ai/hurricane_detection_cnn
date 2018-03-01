import numpy as np
from function import *

path_8="/export/kim79/h2/finished_npy/tracking_data/" # slow and long data: 8
va_input_image=np.load(path_8+"te_input_batch.npy")
w=129;h=86;
d1,d2,d3,d4,d5=np.shape(va_input_image)
va_image=np.reshape(va_input_image[:,:,1:d3,:,:],[d1,d2,d3-1,d4*d5])
va_output_lonlat=np.load(path_8+"te_output_lonlat_batch.npy")
va_lonlat_in=va_output_lonlat[:,:,0:d3-1,:]

image_in=va_image[0]
y_lonlat_in=va_lonlat_in[0]
temp=mask_around_lonlat(image_in,y_lonlat_in)
