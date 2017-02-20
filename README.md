
## Deep Learning Algorithm to mimic human driving behavior

### Kannappan Sirchabesan

### Objective

Develop a deep learning algorithm to mimic human driving behavior.


### Data

#### Data Collection
Use the image data in IMG/ folder and the steering angles provided as a csv file by Udacity.  
Data: https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip

#### Data Loading
Use Pandas read_csv method to load driving_log.csv data.

#### Data Exploration and Visualization
Use Jupyter and matplotlib to explore and visualize the data.

#### Data Augmentation
Augment the data using the below techniques.  
Reference: http://machinelearningmastery.com/image-augmentation-deep-learning-keras/

##### Center Images
Leave the center images and their corresponding steering angles AS IS.

##### Left and Right Images
- Adjust the steering angle output for left and right images.  
- Add a small ratio of 3/25 to the steering angle of left image so that when the model sees a image similar to the left image, it makes a small right angle to the steering angle mentioned for the corresponding center image.  
- Subtract a small ratio of 3/25 to the steering angle of right image so that when the model sees a image similar to the right image, it makes a small left angle to the steering angle mentioned for the corresponding center image.

##### Feature Standardization
Normalization is done as part of the first step in the Keras model, so, ignoring this step.

##### ZCA Whitening
To Do. Might be helpful.

##### Random Rotations
Not essential    
Reason: These are normal road images, so rotations are not essential.

##### Random Shifts
Not essential  
Reason: The left and right images are already being taken into consideration so ignoring this step.

##### Random Flips
Not sure if this will help. Ignoring it for now. Will revisit later if needed.

#### Data Transformation
- Perform RGB -> YUV transformation.  
- Choose a particular Region of Interest from each image.  
- Resize the image to 66x200 to adapt to NVIDIA Network Architecture.  


### Model
#### Network Architecture
Use the End-to-End Deep Learning Model for Self Driving Cars by NVIDIA
https://arxiv.org/abs/1604.07316  
<br><img src="https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/08/cnn-architecture.png" width=300 height=400></br>  

This architecture has been modified to include Dropout layers interspersed between each of the four fully connected layers to handle overfitting.

#### Model Parameters
<table>
<tr><td>Number of Epochs</td><td>2</td></tr>
<tr><td>Samples per Epoch</td><td>28,000</td></tr>  
<tr><td>Validation samples at the end of each epoch</td><td>960</td></tr>
<tr><td>Optimizer</td><td>Adam (0.0001)</td></tr>
<tr><td>Loss Function</td><td>Mean Squared Error (MSE)</td></tr>
<tr><td>Train-Validation Split Ratio</td><td>90:10</td></tr>
</table>

#### Model Saving
Save the model as model.json file  
Save the model weights as model.h5 file

#### Autonomous Car Simulation
Execute the saved model as ```python drive.py model.json``` to simulate the autonomous navigation

#### Results
[![Autonomous Mode](https://img.youtube.com/vi/McJNE4yghC0/0.jpg)](https://www.youtube.com/watch?v=McJNE4yghC0)

---

### Discussion

Tried out transfer learning of some of the architectures like AlexNet and VGG by removing the last layer and replacing with fully connected layer. The vehicle was not driving as well as required and it was a bit more time consuming than the NVIDIA architecture chosen in this project.

##### Model Architecture Design
The Model Architecture of NVIDIA End-to-End deep learning is simple and has only a few layers. The network has 9 layers. The first layer is a normalization layer. The next 5 layers are convolutional layers which are meant for feature extraction from the normalized images. They are followed by 3 fully connected layers.

###### Normalization Layer
This is a single static layer which performs image normalization on the input image

###### Convolutional Layer
* There are 5 convolutional layers that were identified by the NVIDIA engineers through empirical means through various experiments with different layer configurations.
* The first 3 layers have a stride length of 2x2 and a kernel size of 5x5
* The remaining 2 convolutional layers are non-strided with a kernel size of 3x3
* All the layers have a Rectified Linear Unit ReLU non-linear activation

###### Fully Connected Layer
* There are 4 fully connected layers which result in a value that is the inverse turning radius.
* All the 4 layers have a Rectified Linear Unit ReLU non-linear activation

###### Dropout Layer
A Dropout layer has been added to take care of overfitting and make the network a bit more resilient. A Dropout of 0.25 has been interspersed between each of the four fully connected layers.

##### Architecture Characteristics
Some of the good characteristics of this architecture is that it is quite small with few layers and therefore has a lower processing latency. The system is trained end-to-end, so it could be difficult to understand or debug theoretically as to which layers, convolutional or fully connected, are responsible for the various outputs from the network.

##### Data Preprocessing
The quality of data provided by Udacity is quite useful. The images have been resized to 200x66, converted from RGB to YUV and are trained in batches. Augmentation of the data has been performed using the steps mentioned in the link http://machinelearningmastery.com/image-augmentation-deep-learning-keras/ and explained in the previous sections.

##### Model Training (Include hyperparameter tuning.)
Training of the model has been performed on a AWS g2.2xlarge GPU instance. The model is run using the Keras deep learning package. Details of the training are available in [model.ipynb](./model.ipynb) file. Images are trained in batches of 32. Adam Optimizer has been used with a learning rate of 0.0001 after experimenting with a few other values like 0.1, 0.01 and 0.001. Ideally, a grid search on hyperparameters can be performed to identify optimal values for better performance. The training happens via multiple epochs with each epoch having about 28000 samples for training. The data has been divided into training and test sets with a 90:10 ratio after shuffling the entire dataset.
