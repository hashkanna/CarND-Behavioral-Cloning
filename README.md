
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


#### Model Parameters
<table>
<tr><td>Number of Epochs</td><td>2</td></tr>
<tr><td>Samples per Epoch</td><td>10,000</td></tr>  
<tr><td>Validation samples at the end of each epoch</td><td>960</td></tr>
<tr><td>Optimizer</td><td>Adam (0.0001)</td></tr>
<tr><td>Loss Function</td><td>Mean Squared Error (MSE)</td></tr>
<tr><td>Train-Validation Split Ratio</td><td>75:25</td></tr>
</table>

#### Model Saving
Save the model as model.json file  
Save the model weights as model.h5 file

#### Autonomous Car Simulation
Execute the saved model as ```python drive.py model.json``` to simulate the autonomous navigation

#### Results
[![Autonomous Mode](https://img.youtube.com/vi/McJNE4yghC0/0.jpg)](https://www.youtube.com/watch?v=McJNE4yghC0)
