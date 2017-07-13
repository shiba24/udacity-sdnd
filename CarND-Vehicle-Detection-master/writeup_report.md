**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_notcar.png
[image2]: ./output_images/car_feature.png
[image3]: ./output_images/notcar_feature.png
[image4]: ./output_images/sliding_windows.png
[image5]: ./output_images/test_search.png
[image6]: ./output_images/apply_heatmap.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

 ---

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the second code cell of the [IPython notebook](https://github.com/shiba24/udacity-sdnd/blob/master/CarND-Vehicle-Detection-master/script.ipynb).

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters.


#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and finally chose the parameters: using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`, which can be seen in the eighth code cell of the [IPython notebook](https://github.com/shiba24/udacity-sdnd/blob/master/CarND-Vehicle-Detection-master/script.ipynb).

Here are feature images for car and notcar images above.

![alt text][image2]

![alt text][image3]

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Then I trained a linear SVM in the tenth code cell of the [IPython notebook](https://github.com/shiba24/udacity-sdnd/blob/master/CarND-Vehicle-Detection-master/script.ipynb). The accuracy for test dataset (20% of the whole dataset) is `0.989`.


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search windows below the horizontal line and small windows near the horizontal line in the 11-th code cell of the [IPython notebook](https://github.com/shiba24/udacity-sdnd/blob/master/CarND-Vehicle-Detection-master/script.ipynb). The image description of the windows is below:

![alt text][image4]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images on `./test_images/`:

![alt text][image5]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected, then cumulated several succcessive frames. The codes are in the 4th and 5th code cell of the [IPython notebook](https://github.com/shiba24/udacity-sdnd/blob/master/CarND-Vehicle-Detection-master/script.ipynb).

Here are six frames, their corresponding heatmaps and the resulting bounding boxes.

![alt text][image6]


<!--

### Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

 -->