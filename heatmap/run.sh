#! /bin/sh

NAME='nopool_0203'

python 1_train_detection_cnn.py 3ch ${NAME};
python 2_generate_heatmap.py 3ch ${NAME} ;
python 3_pick_centers.py 3ch ${NAME} ;

