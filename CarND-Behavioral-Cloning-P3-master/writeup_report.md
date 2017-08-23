**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/vgg16.png "VGG16"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to feed dataset and to create/train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for 1. feeding data and 2. training the convolution neural network. Feeding data is done in the`MiniBatchLoader` class in the line 26 at the [model.py](https://github.com/shiba24/udacity-sdnd/blob/master/CarND-Behavioral-Cloning-P3-master/model.py). The model definition is in the line 190 at the [model.py](https://github.com/shiba24/udacity-sdnd/blob/master/CarND-Behavioral-Cloning-P3-master/model.py), and the training script is in the line 242 at the [model.py](https://github.com/shiba24/udacity-sdnd/blob/master/CarND-Behavioral-Cloning-P3-master/model.py).
The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.


### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is the modification of VGG16 model that is pre-trained with [IMAGENET](http://www.image-net.org/). Loading the pretrained model is implemented at the line 218 at the [model.py](https://github.com/shiba24/udacity-sdnd/blob/master/CarND-Behavioral-Cloning-P3-master/model.py).

The brief description of the VGG16 network is as following image. (Cited from [this web page](https://www.cs.toronto.edu/~frossard/post/vgg16/))

![alt text][image1]

Then, I chenged the weights and structure of the top layer at the VGG16 network. The structure of the whole network is shown in the next table.

| Layer                 |     Description                               | 
|:---------------------:|:---------------------------------------------:| 
| Input                 | 60x200x3 RGB image                            | 
| VGG16                 | the top layer is not included                 |
| Fully connected       | In: 3072, Out: 512                            |
| ELU                   |                                               |
| BatchNormalization    |                                               |
| Dropout               | KEEP_PROB = 0.5                               |
| Fully connected       | In: 512, Out: 256                             |
| ELU                   |                                               |
| BatchNormalization    |                                               |
| Dropout               | KEEP_PROB = 0.5                               |
| Fully connected       | In: 256, Out: 64                              |
| RELU                  |                                               |
| Dropout               | KEEP_PROB = 0.5                               |
| Fully connected       | In: 64, Out: 1                                |


The data preprocessing including data normalization is done in the `MiniBatchLoader` class.


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 224).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 55 in model.py), which is confirmed by the `MiniBatchLoader` class. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 237).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and the reverse run of the road.

For details about how I created the training data, see the next section. 

```
Train on 17276 samples, validate on 4320 samples
```


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use pre-trained model, and fine-tuning on the augumented dataset. 

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) is the combination of VGG16 model (pre-trained part) and full connected model (fine-tuned part), as described in the previous section.

Here is a summary of the architecture.

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, 60, 200, 3)        0
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 60, 200, 64)       1792
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 60, 200, 64)       36928
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 30, 100, 64)       0
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 30, 100, 128)      73856
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 30, 100, 128)      147584
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 15, 50, 128)       0
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 15, 50, 256)       295168
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 15, 50, 256)       590080
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 15, 50, 256)       590080
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 7, 25, 256)        0
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 7, 25, 512)        1180160
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 7, 25, 512)        2359808
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 7, 25, 512)        2359808
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 3, 12, 512)        0
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 3, 12, 512)        2359808
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 3, 12, 512)        2359808
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 3, 12, 512)        2359808
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 1, 6, 512)         0
_________________________________________________________________
globalaveragepooling2d_1 (Gl (None, 512)               0
_________________________________________________________________
dense_1 (Dense)              (None, 512)               262656
_________________________________________________________________
dropout_1 (Dropout)          (None, 512)               0
_________________________________________________________________
dense_2 (Dense)              (None, 256)               131328
_________________________________________________________________
dropout_2 (Dropout)          (None, 256)               0
_________________________________________________________________
dense_3 (Dense)              (None, 64)                16448
_________________________________________________________________
dropout_3 (Dropout)          (None, 64)                0
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 65
=================================================================
Total params: 15,125,185
Trainable params: 13,389,697
Non-trainable params: 1,735,488
_________________________________________________________________
```


####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
