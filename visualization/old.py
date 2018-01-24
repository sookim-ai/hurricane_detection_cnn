import numpy as np

gt=np.load("ground_trunth_42_1.npy")
pr=np.load("prediction_42_1.npy")
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

gtt=rescale(gt_1);
prr=rescale(pr_1);

gt_lon=[]; gt_lat=[]; pr_lon=[]; pr_lat=[];trj=[];
for i in range(len(gtt)):
     for j in range(len(gtt[0])):
         for k in range(1): 
             if (gtt[i][j][k][0][0]>180.0): 
                 print(str(gtt[i][j][k][0][0])+","+str(gtt[i][j][k][0][1])+","+str(prr[i][j][k][0][0])+","+str(prr[i][j][k][0][1]))
                 gt_lon.append(gtt[i][j][k][0][0]); gt_lat.append(gtt[i][j][k][0][1]); pr_lon.append(prr[i][j][k][0][0]); pr_lat.append(prr[i][j][k][0][1]);

print(np.shape(gt_lon),np.shape(gt_lat),np.shape(pr_lon),np.shape(pr_lat));
#COMPARE WITH TABLE

lon=np.load("lon.npy"); idt=np.load("id.npy");  lat=np.load("lat.npy")
j=0;
for i in range(len(gt_lon)): 
    searching=1;
    while(searching):
        if (abs(gt_lon[i]-lon[i+j])<0.01) and (abs(gt_lat[i]-lat[i+j])<0.01):
            trj.append(idt[i]); 
            searching=0;
            print("found")
        else: 
            j=j+1;

print(np.shape(gt_lon),np.shape(gt_lat),np.shape(pr_lon),np.shape(pr_lat),np.shape(trj));
np.save("gt_lon.npy",gt_lon)
np.save("gt_lat.npy",gt_lat)
np.save("pr_lon.npy",pr_lon)
np.save("pr_lat.npy",pr_lat)
np.save("tid.npy",trj)
