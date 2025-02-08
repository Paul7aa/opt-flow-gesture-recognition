### Description
This system is designed to recognize six distinct gestures in real-time using a 
live camera feed. It employs a two-stream approach, where spatial and temporal features 
are processed separately. These streams are trained independently and later combined using 
a late fusion strategy, enabling accurate gesture recognition by leveraging both appearance 
and motion information. The used architecture is inspired by the work in the paper: 
Simonyan, K., & Zisserman, A. (2014). Two-stream convolutional networks for action 
recognition in videos.

### Dataset 
The dataset used for training the model is composed of 1050 self-recorded videos, with 
frame counts varying from 25 to 80 frames. These videos were recorded and labeled using 
a simple script record_dataset.py. This multi-class classification problem aims to classify 
the following labels: 
```
LABELS = { 
0 : "Swipe Left",  
1: "Swipe Right",  
2: "Swipe Up",  
3 : "Swipe Down",  
4: "Zoom In", 
5: "Zoom Out", 
6: "Doing nothing" 
}
```

The addition of the “Doing nothing” class is justified by the need to allow the model to 
distinguish between meaningful gestures and moments where no gesture is performed. 
The videos in the dataset are used to train both the spatial and temporal stream of the 
model.

### Model structure and training 
The trained model is formed of two separate streams:

* Spatial stream 

This stream takes a single image from the video as input as to obtain important 
details such as the presence of the hand, whether it’s flat (as in the case of swipes) or 
slightly clenched (as in the case of zoom in / zoom out). By focusing on these spatial 
cues, the stream complements the temporal stream by providing frame-specific 
details about the hand’s shape and position. This stream proves to ensure better 
confidence in the model’s classifications.
When training the spatial stream, a random frame from the videos was selected to 
ensure the resulting model is robust and can distinguish specific gestures without the 
user having completed the movement. 
The images fed into the spatial stream for training are first resized to 224 x 224 
(standard dimension for CNNs). After, augmentations are randomly applied and the 
result is normalized using z-score normalization. The means and variances used for 
normalization are the ones used for the ImageNet dataset. These parameters were 
used to improve convergence. 
The stream itself represents a standard CNN comprised of 5 convolutional layers, batch normalization, 
max pooling layers and 2 fully-connected layers are used for the spatial 
stream. This stream has a small number of 1.7 million parameters, which is 
enough to capture relevant information. 
The input shape used for this stream is ``` (batch_size, 3, 224, 224) ```.

* Temporal stream

The temporal stream aims to distinguish specific gestures using partial dense optical
flow sequences from the input video. A fixed number of  30 optical-flow frames was 
used to train this stream. Given this constraint, the approach to training this stream is to 
feed it 30 consecutive optical-flow frames from the input video. Before computing 
optical flow, the images are resized to 224 x 224. 
During hyperparameter tuning, it was discovered that using raw optical-flow achieves 
better results compared to magnitude and angle extraction from the frames. This could be 
due to the preservation of detailed motion information, representing both direction (angle) 
and intensity (magnitude) in a more granular form. 
Given the nature of the data, a more complex model had to be trained in order to 
capture information from the optical-flow frames. The used architecture 
levarages 3D convolution layers to process the sequences of frames. It is 
comprised of 5 3D convolution layers, batch normalization and just a few max 
pooling layers to ensure it the stream doesn’t downscale the temporal dimension 
too aggresively. The features are then fed into two linear layers for classification. 
This stream has a total of around 5.6 million parameters and takes input of shape 
``` (batch_size, 2, 30, 224, 224) ```.

### Hyperparameter Tuning
The hyperparameter tuning process played a pivotal role in optimizing the performance of 
the gesture recognition model. Given the intricate nature of the data and the computational 
costs associated with training, a well-defined search strategy was essential. The process 
began by defining a search space that included parameters for the Farneback optical flow 
algorithm and the magnitude threshold used for motion filtering. These parameters directly 
influenced the quality and informativeness of the optical flow representations fed into the 
temporal stream of the model.  
For instance, parameters such as pyr_scale, levels, winsize, and poly_sigma controlled the 
level of detail captured in the optical flow, while the magnitude threshold determined the 
sensitivity of motion detection. The used search space was: 

```
search_space = { 
    "magnitude_thresh": [1, 2.5], 
    "farneback_params": { 
        "pyr_scale": [0.3, 0.4, 0.5], 
        "levels": [1, 2, 3, 4], 
        "winsize": [9, 15], 
        "iterations": [3, 4, 5], 
        "poly_n": [5, 7], 
        "poly_sigma": [1.1, 1.2], 
        "flags": [0], 
        }, 
    } 
```

Using random search, the best parameters found were the following: 

```
farneback_params = { 
    "magnitude_thresh": 1, 
    "farneback_params": { 
        "pyr_scale": 0.4, 
        "levels": 4, 
        "winsize": 15, 
        "iterations": 5, 
        "poly_n": 5, 
        "poly_sigma": 1.2, 
        "flags": 0, 
        }, 
    } 
```

### Inference
The inference pipeline processes live camera input to classify gestures in real-time using a 
two-stream model. The system captures RGB frames and computes optical flow between 
consecutive frames, storing the motion vectors. Once enough optical flow frames are 
collected, the temporal stream analyzes motion patterns, while the spatial stream processes 
the most recent RGB frame to extract static features. 
The outputs of both streams are combined using late fusion, prioritizing the temporal 
stream for motion analysis. Predictions are made periodically, with class probabilities 
determining the gesture. A confidence threshold ensures reliability, and predictions are 
displayed on the video feed. This pipeline effectively combines spatial and temporal 
information for real-time gesture recognition.

![image](https://github.com/user-attachments/assets/5b92a2eb-d5a1-4fe1-b631-996f75d3b36e)
![image](https://github.com/user-attachments/assets/647b2519-6046-413f-ac20-ebce48bf2867)
![image](https://github.com/user-attachments/assets/671372ae-e123-4797-bc0f-02ccb01c3ec8)
![image](https://github.com/user-attachments/assets/20bb9861-cbd2-4f01-804a-8055a4a61777)
![image](https://github.com/user-attachments/assets/96cd7497-dc0d-4078-b713-39e62bd5fbc8)
![image](https://github.com/user-attachments/assets/25482041-353a-43f8-8cd0-3b7cc6c7b020)
![image](https://github.com/user-attachments/assets/1284bdcb-5d83-4131-8b90-0db7f5286f30)
