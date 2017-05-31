**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/class_distribution.png "Class Distribution"
[image2]: ./examples/original.png "Original"
[image3]: ./examples/grayscale.png "Gray Scale"
[image4]: ./examples/augmentation.png "Data Augmentation"
[image5]: ./examples/test_signs.png "Traffic Signs"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/shiba24/udacity-sdnd/blob/master/CarND-Traffic-Sign-Classifier-Project-master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is `len(X_train) = 34799`
* The size of the validation set is `len(X_valid) = 12630`
* The size of test set is `len(X_test) = 12630`
* The shape of a traffic sign image is `X_train.shape[1:] = (32, 32, 3)`
* The number of unique classes/labels in the data set is `len(list(set(y_train))) = 43`

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the distribution of each class in the training dataset. Basicaly it is better that every class has similar number of images when training models. From the figure, we can see the distribution is not much constant. Hence it can be said that to balance the dataset (add samples of rare classes) would improve the accuracy in the future.

![alt text][image1]

### Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As preprocessing, I normalized the image data. It is always good to normalize images to avoid divergent of the output of neural network. Here are examples of original and gray-scaled images.

![alt text][image2]

![alt text][image3]


I decided to generate additional data because of unbalance of classes, as we saw in the section 2.

To augment the dataset, I used scaling up and down the images randomly. Here is an example of an original image and an augmented image:

![alt text][image4]

When we decide a maximum coefficient to scaling, the function `scaling` set the actual value of scaling from `-1 * coef` to `coef`. The value larger than 0 means scaling up and the value smaller than 0 means scaling down. When scaling up the images, the scaled image will be cropped the center. Otherwise the image will be padded with zero.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer                 |     Description                               | 
|:---------------------:|:---------------------------------------------:| 
| Input                 | 32x32x3 RGB image                             | 
| Convolution 5x5       | 1x1 stride, same padding, outputs 16x16x16    |
| RELU                  |                                               |
| Max pooling           | 2x2 stride,  outputs 14x14x16                 |
| Convolution 5x5       | 1x1 stride, same padding, outputs 7x7x32      |
| RELU                  |                                               |
| Max pooling           | 2x2 stride,  outputs 5x5x16                   |
| Convolution 3x3       | 1x1 stride, same padding, outputs 3x3x64      |
| RELU                  |                                               |
| Fully connected       | In: 576, Out: 256                             |
| RELU                  |                                               |
| Dropout               | KEEP_PROB = 0.5                               |
| Fully connected       | In: 256, Out: 128                             |
| RELU                  |                                               |
| Fully connected       | In: 128, Out: 43                              |
| Softmax               |                                               |
 

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used Adam as optimizer. The loss function is cross entropy. All the parameters in training are as follows:

```
EPOCHS = 25
BATCH_SIZE = 64
SCALE = 0.1
rate = 0.0004
```

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 98.6%
* validation set accuracy of 95.2%
* test set accuracy of 93.5%

I did an iterative approach:
* What was the first architecture that was tried and why was it chosen?
-- I chose AlexNet and improved it.
* What were some problems with the initial architecture?
-- Less accurate.
* How was the architecture adjusted and why was it adjusted?
-- Adding dropout (critical), and adding one convolutional layer.
* Which parameters were tuned? How were they adjusted and why?
-- Learning rate and Scale value were tuned, because 1. decreasing learning rate almost always works well, 2. Scaking augmentation was only applied to training dataset, which means scaling value should not be too large.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
-- In this task, dropout was critical. Practically we can avoid overfitting by adding dropout layes. Of course, convolutional layers are good since the input data is images and pooling layers are also good since the task is classification.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![alt text][image5]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the 14th cell of the Ipython notebook.

Here are the results of the prediction:

| Image                 |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (70km/h)  | Speed limit (70km/h)                          | 
| Speed limit (100km/h) | Roundabout mandatory                          |
| Speed limit (30km/h)  | Speed limit (30km/h)                          |
| No stopping           | Traffic signals                               |
| Stop                  | Stop                                          |
| Right of way          | Right of way                                  |


The model was able to correctly guess 4 of the 6 traffic signs, which gives an accuracy of 67%. This is a bit worse than the accuracy on the test set of 93.5%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located from the 17th cell of the Ipython notebook.

 - The top five soft max probabilities of Speed limit (70km/h)

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| 0.999995              | __Speed limit (70km/h)__                      | 
| 4.74793e-06           | Speed limit (20km/h)                          |
| 4.50994e-09           | Speed limit (60km/h)                          |
| 1.0691e-10            | Speed limit (120km/h)                         |
| 6.74825e-12           | Ahead only                                    |

 - The top five soft max probabilities of Speed limit (100km/h)

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
|  0.990586             | Roundabout mandatory                          | 
|  0.00904871           | __Speed limit (100km/h)__                     |
|  0.000188292          | Slippery road                                 |
|  9.07689e-05          | Vehicles over 3.5 metric tons prohibited      |
|  4.72472e-05          | Dangerous curve to the left                   |

 - The top five soft max probabilities of Speed limit (30km/h)

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
|  1.0                  | __Speed limit (30km/h)__                      | 
|  1.82738e-12          | Speed limit (50km/h)                          |
|  2.80644e-15          | Speed limit (80km/h)                          |
|  4.89327e-20          | Speed limit (20km/h)                          |
|  6.01662e-23          | Stop                                          |

 - The top five soft max probabilities of No stopping

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
|  0.801144             | Traffic signals                               | 
|  0.0895388            | Keep right                                    |
|  0.0500171            | Road work                                     |
|  0.029327             | Priority road                                 |
|  0.0176616            | Turn left ahead                               |

 - The top five soft max probabilities of Stop

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
|   1.0                 | __Stop__                                      | 
|  1.32871e-22          | No entry                                      |
|   2.33511e-30         | Speed limit (60km/h)                          |
|   1.33455e-32         | Bicycles crossing                             |
|   1.35874e-34         | Speed limit (80km/h)                          |


 - The top five soft max probabilities of Right-of-way at the next intersection

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
|  1.0                  | __Right-of-way at the next intersection__     | 
|  3.63465e-08          | Beware of ice/snow                            |
|  2.79826e-14          | Pedestrians                                   |
|  8.21564e-18          | Double curve                                  |
|  2.541e-19            | Road work                                     |


We can see that in the case of correct prediction, the model has the large confidence (almost 1.0 probability). 

