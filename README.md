DATASET:

The dataset can be also downloaded from the following link:
https://visionlab.uncc.edu/download/summary/60-data/477-ucf-anomaly-detection-dataset


You can also download dataset in parts through following link

https://www.dropbox.com/sh/75v5ehq4cdg5g5g/AABvnJSwZI7zXb8_myBA0CLHa?dl=0


Below you can find Training/Testing Code for our anomaly Detection project which was published in Computer Vision and Pattern Recognition, CVPR 2018.

The implementation is tested using:

Keras version 1.1.0

Theano 1.0.2

Python 3

Ubuntu 16.04


We used C3D-v1.0 (https://github.com/facebook/C3D) with default settings as a feature extractor.
 
Training_AnomalyDetecor_public.py is to Train Anomaly Detection Model


Testing_Anomaly_Detector_public.py is to test trained Anomaly Detection Model


Save_C3DFeatures_32Segments is to save already computed C3D features for the whole video into 32 segment features.


weights_L1L2.mat: It contains the pre-trained weights for the model ‘model.json’.

Demo_GUI: We have provided a simple GUI which can be used to see results of our approach on sample videos.

SampleVideos: This folder contains C3D features (32 segments) for sample videos. It order to see testing results for the features in this folder, please copy the corrosponding videos in the same folder.


Plot_All_ROC:  This code can be use to plot the ROC results reported in the paper. The data to plot ROCs of methods discussed in the paper can be found in folder Paper_Results.


The project page can be found at: http://crcv.ucf.edu/projects/real-world/

Temporal_Anomaly_Annotation.txt contains ground truth annotations of the testing dataset.

Anomaly_Train.txt contains the video names for training anomaly detector

FAQs: 


Q:  Should I use C3D or I3D?


Ans:  Several people have emailed me that in their experiments, I3D performs much better than C3D. So I would suggest to first try I3D. Obviously, for this, we need to re-train the model and make small modifications in training and testing codes.



If you find any bug, or have some questions, please contact Waqas Sultani: waqas5163@gmail.com


Citation:

@InProceedings{Sultani_2018_CVPR,

author = {Sultani, Waqas and Chen, Chen and Shah, Mubarak},

title = {Real-World Anomaly Detection in Surveillance Videos},

booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},

month = {June},

year = {2018}

}
