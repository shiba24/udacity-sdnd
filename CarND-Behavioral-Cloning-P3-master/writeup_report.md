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
[image5]: ./images/test4.png "Original Image"
[image6]: ./images/test4_hue.png "Hue-changes Image"
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

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and the reverse run of the road. As data augumentation process, I implemented fliplr, changing hue processes. For details about how I created and augumented the training data, see the next section. 


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use pre-trained model, and fine-tuning on the augumented dataset. This idea is from the fact that input is image, which means the model trained on other images already would share some parameters like wights and biases. VGG16 network achieves 92.7% top-5 test accuracy in ImageNet, which is a dataset of over 14 million images belonging to 1000 classes. Then I changed the weights and biases of the top-layer of VGG16 network, and add several full-connected layers in order to fine-tune the model.

To combat the overfitting, I augumented the dataset randomly and dynamically. This is possible by using `iterator` to feed minibatch to the model. The augument method I used were flipping images and changing hue.

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


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn how to get back from the side to the center position. Here are left-side and right-side images:

![alt text][image3]
![alt text][image4]

In addition, I recorded two laps of the track for reverse run of the road, because the original track is counterclockwise rotation and that would result in the left-shifted prediction of the neural network, which is not expected.

Before training, I  randomly shuffled the data set and splitted the dataset into training (90%) and validation (10%). Finally, the number of the original dataset (not augumented yet) was as follows.

```
Train on 17276 samples, validate on 4320 samples
```

To augment the data sat, I did 1. changing hues of the images randomly, 2. flipping images and angles.

Changing hues (line 159 in model.py) purely augument the dataset, because I noticed there were several hue environment while drinving the road. Here are the code and examples of original image and hue-changed image.

```
    def change_hue(self, img, delta_hue):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[:, :, 0] += delta_hue
        hued_img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        return hued_img
```

![alt text][image5]
![alt text][image6]

Flipping images would augument the dataset, and also the distribution of the steering angle would be less unbalanced to counterclockwise, which is because the track is counterclockwise rotation. Here are the code and an image that has then been flipped:


```
    def fliplr(self, minibatch_X, minibatch_y):
        def flip(img, flag):
            if flag == -1:
                return cv2.flip(img, 1)
            else:
                return img
        flipflag = np.random.choice((-1, 1), size=(len(minibatch_X), ))   # -1... flip  /  1...not flip
        flipped_X = np.array([flip(minibatch_X[i], flipflag[i]) for i in range(0, len(minibatch_X))])
        flipped_y = flipflag * minibatch_y
        return flipped_X, flipped_y
```


The `MiniBatchLoader` class, which is iterator, dynamically and randomly applys these augumentation processing randomly (line 92 in model.py). To use the `MiniBatchLoader` iterator, I used `model.fit_generator()` method of keras. The images are standardized (line 176) and reshaped, cropped the bottom part (line 138). The input image shape was (60, 200).

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 30 as evidenced by error rate . I used an adam optimizer so that manually training the learning rate wasn't necessary.
