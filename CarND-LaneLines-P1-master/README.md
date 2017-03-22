**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/mask.png "Mask"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps below.
 1. Convert the image to grayscale
 2. Apply gaussian blur to the gray image
 3. Detect edges by canny algorithm
 4. Mask half low triangle of the image (focusing on gray part in the image)
 ![masking][image1]
 5. Combine egdes to lines using hough transform

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by three processes.

 1. Decide each line candidate is in which (right or left) part in the image
 2. Calculate average line for right lines and left lines, respectivelty
 3. Extend the two lines to the bottom of the image

You can see all the results of applying the algorithms to test images [here](https://github.com/shiba24/udacity-sdnd/tree/master/CarND-LaneLines-P1-master/test_images).

### 2. Identify potential shortcomings with your current pipeline

One potential shortcoming would be, essentially, it is hyper-tuned with the test datasets. Hence we even don't know whether this hyperparameters will work well or not for new images (roads). For example, `min_line_length` may be affected by how much the road is curving.

### 3. Suggest possible improvements to your pipeline

One possible way to choose better hyperparameters. Hyperparameter search might be not essential, but is important for the _last one_ algorithm improvement. For searching, we can use some libraries like [hyperopt](https://github.com/hyperopt/hyperopt).

Another possible improvement would be to make lines smooth among frames in video detection. This would be possible by having the detected line of past frames in memory.

Another potential improvement could be to use color information. Actually lane lines tend to be white or yellow (not blue, green) in general. But converting images to grayscale might make the information less useful.

The last one is improvement of the algorhithm itself. If we use Convolutional Neural network, the resutl will be improved (maybe as we will see later).
