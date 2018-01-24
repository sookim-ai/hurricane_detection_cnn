import numpy as np

gt=np.load("ground_trunth_401.npy")
pr=np.load("prediction_401.npy")
gt_1=gt[:,:,:,0:1,:]
pr_1=pr[:,:,:,0:1,:]


def rescale(gt_1):
    for i in range(len(gt_1)):
        for j in range(len(gt_1[0])):
            for k in range(len(gt_1[0][0])):
                for h in range(len(gt_1[0][0][0])):
                    lon=gt_1[i][j][k][h][0];
                    lat=gt_1[i][j][k][h][1];
                    lat=80.0*lat;
                    lon=lon*160.0+180.0;
                    gt_1[i][j][k][h][0]=lon;
                    gt_1[i][j][k][h][1]=lat;
    return gt_1;

def move(source, target):
    for i in range(len(source)):
        target.append(source[i]);

def move_n(source, target,n):
    for i in range(n):
        target.append(source[i]);


gtt=rescale(gt_1);
prr=rescale(pr_1);
gt_lon=[]; gt_lat=[]; pr_lon=[]; pr_lat=[];trj=[];
for i in range(len(gtt)):
     j=0; 
     while (j<len(gtt[0])): #24 batch
         count=0;
         gt_lon_temp=[]; gt_lat_temp=[]; pr_lon_temp=[]; pr_lat_temp=[];
         for k in range(len(gtt[0][0])): #24 ts 
             if (gtt[i][j][k][0][0]>180.0): 
                 #print(str(gtt[i][j][k][0][0])+","+str(gtt[i][j][k][0][1])+","+str(prr[i][j][k][0][0])+","+str(prr[i][j][k][0][1]))
                 gt_lon_temp.append(gtt[i][j][k][0][0]); gt_lat_temp.append(gtt[i][j][k][0][1]); pr_lon_temp.append(prr[i][j][k][0][0]); pr_lat_temp.append(prr[i][j][k][0][1]);
                 count=count+1;
         j=j+count;
         print(str(i)+" "+str(j)+" "+str(count)+" "+str(len(gt_lon_temp))+"\n");
         if (j<24):
             move(gt_lon_temp,gt_lon);
             move(gt_lat_temp,gt_lat);
             move(pr_lon_temp,pr_lon);
             move(pr_lat_temp,pr_lat);
         else:
             n=24-(j-count);
             move_n(gt_lon_temp,gt_lon,n);
             move_n(gt_lat_temp,gt_lat,n);
             move_n(pr_lon_temp,pr_lon,n);
             move_n(pr_lat_temp,pr_lat,n);
#ii=[202.651,211.396,211.601,261.129,261.723,254.973,266.178,264.382,273.707];
#ii_l = [ '%.2f' % elem for elem in ii ] 
#lonn = [ '%.2f' % elem for elem in gt_lon ] 
#for k in range(len(ii_l)):
#    iiv=lonn.index(ii_l[k])
#    del gt_lon[iiv];
#    del gt_lat[iiv];
#    del pr_lon[iiv];
#    del pr_lat[iiv];

for i in range(len(gt_lon)):
    print(str(i)+","+str(gt_lon[i])+","+str(gt_lat[i])+","+str(pr_lon[i])+","+str(pr_lat[i]))
print(np.shape(gt_lon),np.shape(gt_lat),np.shape(pr_lon),np.shape(pr_lat));
#COMPARE WITH TABLE

#lon=np.load("lon.npy"); idt=np.load("id.npy");  lat=np.load("lat.npy")
#j=0;
#for i in range(len(gt_lon)): 
#    searching=1;
#    while(searching):
#        if (abs(gt_lon[i]-lon[i+j])<0.01) and (abs(gt_lat[i]-lat[i+j])<0.01):
#            trj.append(idt[i]); 
#            searching=0;
#            print("found",i)
#        else: 
#            j=j+1;

#lonn = [ '%.2f' % elem for elem in lon ] #round up
#gt_lonn=[ '%.2f' % elem for elem in gt_lon ]

#for i in range(len(gt_lonn)):
#    trj.append(idt[list(lonn).index(gt_lonn[i])])
            
print(np.shape(gt_lon),np.shape(gt_lat),np.shape(pr_lon),np.shape(pr_lat));
np.save("gt_lon.npy",gt_lon)
np.save("gt_lat.npy",gt_lat)
np.save("pr_lon.npy",pr_lon)
np.save("pr_lat.npy",pr_lat)
#np.save("tid.npy",trj)
