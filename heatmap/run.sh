#! /bin/sh

NAME='nopool_0208'

#python 1_train_detection_cnn.py 2ch ${NAME};
python 2_generate_heatmap.py 2ch ${NAME} ;
# python 3_pick_centers.py 2ch ${NAME} ;

