
# Climate Deep Learning Code (optimized for Tensorflow 1.4)


1. Heatmap: This folder contains following models to detect hurricane using CNN. (By sangwoong yoon)

   (1) detection CNN (1_train_detection_cnn.py) : 
   
      The model to detect extra-tropical cyclone in multi-channeled climate data with 98.2% accuracy.
      Once you prepare your own dataset  (I can share dataset only to request)
   
       >> python 1_train_detection_cnn.py
         
   (2) The model to generate heatmap (2_generate_heatmap.py) :

      This model generate heatmap using detection CNN by convolute learned feature from detection CNN through global scaled         climate data.

       >> python 2_generate_heatmap.py
    
   (3) Showcase: visualize_heatmap.ipynb
   
2. Tracking Network: This folder contains following models to track hurricane using RNN. (By sookyung kim)

   There are three main components: 
      (1) detecting extreme climate events using CNNs, 
      (2) generating a heat map based on the    feature we found with detection CNN, and 
      (3) tracking trajectories from heat map using LSTM networks. 

   First, convolutional neural networks are trained to detect extreme climate event, 
   and learn a distinctive feature of event from massive-scaled cropped climate re-analysis data. 
   Our architecture is 2-layered CNNs. Specifically, each layer contains one convolutional layer and one pooling layer.       First convolutional layer has 32 features with 5 x 5 kernels and second convolutional layer has 64 features with 7 x 7 kernels. 
   We use max-pooling with size of 2 x 2. We apply dropout~ to exclude 20% of neurons in order to reduce over-fitting.
   Lastly, one dense layer with ReLU activation and one fully connected layer is followed. We use softmax 
   The output has one class representing probability of existing event in input and a softmax activation has been used.
   
   (1) exact_position:
         Output of RNN is exact location of longitude and latitude of hurricane center in 2d-input image.
            
           >> python main_v1.py
         
   (2) grad_position:
         Output of RNN is difference of location of longitude and latitude of hurricane center between input of previous time-step and input of current time step. 
         
            >> python main_v1.py
    
    
    

