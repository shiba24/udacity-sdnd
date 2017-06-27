**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/calibration_undistortion.png "Undistorted"
[image2]: ./output_images/pipeline_image.png "Road Transformed"
[image3]: ./output_images/binary_image.png "Binary Example"
[image4]: ./output_images/warp_perspective.png "Warp Example"
[image5]: ./output_images/lane_pixels.png "Lane pixels"
[image6]: ./output_images/polyfit.png "Polyfit lane line"
[video1]: ./project_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the [IPython notebook](https://github.com/shiba24/udacity-sdnd/blob/master/CarND-Advanced-Lane-Lines-master/script.ipynb). 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function.

This is a result of camera calibration and undisgtortion on an test image picked up from ```project_video.mp4```.

![alt text][image1]

### Pipeline (single images)

Here I describe how to detect lane lines on single image, which will be easy to apply to the video ```project_video.mp4```

Before explaining the pipeline in detail, I show you the lane-detected image after all the process of the pipeline. This frame is the same one as the former one.

![alt text][image2]

#### 1. Provide an example of a distortion-corrected image.

I have already described how I applied the distortion correction to one of the test images in [Camera Calibration section](https://github.com/shiba24/udacity-sdnd/blob/master/CarND-Advanced-Lane-Lines-master/writeup_report.md#camera-calibration).


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image in the 3rd code cell of the [IPython notebook](https://github.com/shiba24/udacity-sdnd/blob/master/CarND-Advanced-Lane-Lines-master/script.ipynb).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`, which appears in the 4th code cell of the [IPython notebook](https://github.com/shiba24/udacity-sdnd/blob/master/CarND-Advanced-Lane-Lines-master/script.ipynb).  The `warp()` function takes as inputs an image (`img`).  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I detected lane lines by finding peaks in the image. These are done from the 5th to 8th code cell of the [IPython notebook](https://github.com/shiba24/udacity-sdnd/blob/master/CarND-Advanced-Lane-Lines-master/script.ipynb).

![alt text][image5]

After that, I fit my lane lines with a 2nd order polynomial. 

![alt text][image6]


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in the 7th code cell of the [IPython notebook](https://github.com/shiba24/udacity-sdnd/blob/master/CarND-Advanced-Lane-Lines-master/script.ipynb).

```
def measuring_curv(l_x, l_y, r_x, r_y):
    l_max = 720
    r_max = 720
    ym_per_pix = 30./720           # meters per pixel in y dimension
    xm_per_pix = 3.7/700           # meteres per pixel in x dimension

    left_fit_cr = np.polyfit(l_y * ym_per_pix, l_x * xm_per_pix, 2)
    right_fit_cr = np.polyfit(r_y * ym_per_pix, r_x * xm_per_pix, 2)
    left_curverad = ((1 + (2 * left_fit_cr[0] * l_max + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * r_max + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])
    average_curv = (left_curverad + right_curverad) / 2
    return average_curv

def get_text_info(img, l_x, l_y, r_x, r_y, l_lane_pix, r_lane_pix):
    xm_per_pix = 3.7/700             # meteres per pixel in x dimension
    screen_middel_pixel = img.shape[1] / 2
    car_middle_pixel = int((r_lane_pix + l_lane_pix)/2)
    screen_off_center = screen_middel_pixel - car_middle_pixel
    meters_off_center = round(xm_per_pix * screen_off_center, 2)
    curv_in_meters = int(measuring_curv(l_x, l_y, r_x, r_y))
    return meters_off_center, curv_in_meters
```


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.


![alt text][image2]


---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_output.mp4)

