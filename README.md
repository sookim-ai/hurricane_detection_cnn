
# Climate Deep Learning Code (optimized for Tensorflow 1.4)


1. Heatmap: This folder contains following mode. (By sangwoong yoon)

(1) detection CNN (1_train_detection_cnn.py) : 

The model to detect extra-tropical cyclone in multi-channeled climate data with 98.2% accuracy.
Once you prepare your own dataset  (I can share dataset only to request)
   
    >> python 1_train_detection_cnn.py
 
(2) The model to generate heatmap (2_generate_heatmap.py) :

This model generate heatmap using detection CNN by convolute learned feature from detection CNN through global scaled       climate data.

    >> python 2_generate_heatmap.py
    
(3) Showcase: visualize_heatmap.ipynb
    
