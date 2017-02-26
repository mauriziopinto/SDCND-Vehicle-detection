# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


In this project, the goal is to write a software pipeline to detect vehicles in a video.


**Vehicle Detection Project**

---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector. 
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the pipeline on a video stream (test_video.mp4 and project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train the classifier.  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.

Some example images for testing your pipeline on single frames are located in the `test_images` folder.  The output images from each stage of the pipeline are saved in the folder called `ouput_images`, and are included in this document.


[//]: # (Image References)
[image1]: ./datasets/vehicles/GTI_MiddleClose/image0089.png
[image2]: ./datasets/non-vehicles/GTI/image52.png
[t1]: ./test_images/test1.jpg
[w1]: ./output_images/hotwindows/test1.jpg.png
[h1]: ./output_images/heatmap/test1.jpg.png
[l1]: ./output_images/labels/test1.jpg.png
[f1]: ./output_images/final/test1.jpg.png
[t2]: ./test_images/test2.jpg
[w2]: ./output_images/hotwindows/test2.jpg.png
[h2]: ./output_images/heatmap/test2.jpg.png
[l2]: ./output_images/labels/test2.jpg.png
[f2]: ./output_images/final/test2.jpg.png
[t3]: ./test_images/test3.jpg
[w3]: ./output_images/hotwindows/test3.jpg.png
[h3]: ./output_images/heatmap/test3.jpg.png
[l3]: ./output_images/labels/test3.jpg.png
[f3]: ./output_images/final/test3.jpg.png
[t4]: ./test_images/test4.jpg
[w4]: ./output_images/hotwindows/test4.jpg.png
[h4]: ./output_images/heatmap/test4.jpg.png
[l4]: ./output_images/labels/test4.jpg.png
[f4]: ./output_images/final/test4.jpg.png
[t5]: ./test_images/test5.jpg
[w5]: ./output_images/hotwindows/test5.jpg.png
[h5]: ./output_images/heatmap/test5.jpg.png
[l5]: ./output_images/labels/test5.jpg.png
[f5]: ./output_images/final/test5.jpg.png
[t6]: ./test_images/test6.jpg
[w6]: ./output_images/hotwindows/test6.jpg.png
[h6]: ./output_images/heatmap/test6.jpg.png
[l6]: ./output_images/labels/test6.jpg.png
[f6]: ./output_images/final/test6.jpg.png
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

The README for this project is available on [Github](https://github.com/mauriziopinto/SDCND-Vehicle-detection/blob/master/README.md)

All the code used is in the file main.py, available at the same Github repository.

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The HOG features are extracted in the `get_hog_features` function:

```python
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
					 vis=False, feature_vec=True):
	# Call with two outputs if vis==True
	if vis == True:
		hogdescripter = cv2.HOGDescriptor()
		hogdescripter.compute(img)
		features, hog_image = hog(img, orientations=orient,
								  pixels_per_cell=(pix_per_cell, pix_per_cell),
								  cells_per_block=(cell_per_block, cell_per_block),
								  transform_sqrt=True,
								  visualise=vis, feature_vector=feature_vec)
		return features, hog_image
	# Otherwise call with one output
	else:
		features = hog(img, orientations=orient,
					   pixels_per_cell=(pix_per_cell, pix_per_cell),
					   cells_per_block=(cell_per_block, cell_per_block),
					   transform_sqrt=True,
					   visualise=vis, feature_vector=feature_vec)
		return features
```  

The `get_hog_features` function relies on the cv2.HOGDescriptor functionality: all the images in the folders datasets/vehicles and datasets/non-vehicles have been processed with the `get_hog_features` method in order to train a linear SVM classifier (LinearSVC)

Here is an example of one of each of the `vehicle`(8792 samples) and `non-vehicle` (8968 samples) classes:

| Vehicles | Non-vehicles | 
| ------------- |:-------------:| 
| ![Vehicle][image1] | ![Non-vehicle][image2]|

The training set has two balanced classes, so we don't risk to overfit to the majority class.

The core of the project is in the `detect_vehicles` function: before images are processed for the extraction of the HOG features, some previous processing steps (color space conversion, resize) are executed:

```python
img = img.astype(np.float32)/255
	
img_tosearch = img[ystart:ystop,:,:]
ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
if scale != 1:
	imshape = ctrans_tosearch.shape
	ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
	
ch1 = ctrans_tosearch[:,:,0]
ch2 = ctrans_tosearch[:,:,1]
ch3 = ctrans_tosearch[:,:,2]

# Define blocks and steps as above
nxblocks = (ch1.shape[1] // pix_per_cell)-1
nyblocks = (ch1.shape[0] // pix_per_cell)-1 
nfeat_per_block = orient*cell_per_block**2
# 64 was the original sampling rate, with 8 cells and 8 pix per cell
window = 64
nblocks_per_window = (window // pix_per_cell)-1 
cells_per_step = 2  # Instead of overlap, define how many cells to step
nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
nysteps = (nyblocks - nblocks_per_window) // cells_per_step

# Compute individual channel HOG features for the entire image
hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
```
Images are first converted from the RGB colorspace to the YCrCb one and then the HOG features extraction is applied to the entire image (once for each channel). This choice improved the overall performances of the application, because it avoided to extract HOG features for all of the small windows created during the sliding window stage.


####2. Explain how you settled on your final choice of HOG parameters.

There are many parameters that can influence the HOG feature extraction:

* color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
* orient = 9  # HOG orientations
* pix_per_cell = 8  # HOG pixels per cell
* cell_per_block = 2  # HOG cells per block
* hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
* spatial_size = (16, 16)  # Spatial binning dimensions
* hist_bins = 16  # Number of histogram bins
* spatial_feat = True  # Spatial features on or off
* hist_feat = True  # Histogram features on or off
* hog_feat = True  # HOG features on or off

I tried many combinations on the video "test_video.mp4" and selected a combination of parameters that provide sufficiently good results for that video. I then used the same parameters for the project video.


####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

A linear SVM model is built in the `train_svm()` function:

```python
def train_svm():
	"""
	Train a SVM model and save the model and the scaler on disk

	Args:
		
	Returns:
		
	"""
	carFiles = glob.glob("datasets/vehicles/*/*.png", recursive=True)
	print("Loading car files")
	cars = []
	for image in carFiles:
		cars.append(image)
	print("Loaded car files")
	nonCarFiles = glob.glob("datasets/non-vehicles/*/*.png", recursive=True)
	notcars = []
	print("Loading non-car files")
	for image in nonCarFiles:
		notcars.append(image)

	print("Loaded" , len(cars) , "cars")
	print("Loaded" , len(nonCarFiles) , "non cars")
	print("Extracting car features")
	car_features = extract_features(cars, color_space=color_space,
									spatial_size=spatial_size, hist_bins=hist_bins,
									orient=orient, pix_per_cell=pix_per_cell,
									cell_per_block=cell_per_block,
									hog_channel=hog_channel, spatial_feat=spatial_feat,
									hist_feat=hist_feat, hog_feat=hog_feat)
	print("Extracting non-car features")
	notcar_features = extract_features(notcars, color_space=color_space,
									   spatial_size=spatial_size, hist_bins=hist_bins,
									   orient=orient, pix_per_cell=pix_per_cell,
									   cell_per_block=cell_per_block,
									   hog_channel=hog_channel, spatial_feat=spatial_feat,
									   hist_feat=hist_feat, hog_feat=hog_feat)
	X = np.vstack((car_features, notcar_features)).astype(np.float64)
	# Fit a per-column scaler
	X_scaler = StandardScaler().fit(X)
	# Apply the scaler to X
	scaled_X = X_scaler.transform(X)
	# Define the labels vector
	y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
	# Split up data into randomized training and test sets
	rand_state = np.random.randint(0, 100)
	X_train, X_test, y_train, y_test = train_test_split(
		scaled_X, y, test_size=0.3, random_state=rand_state)
	print('Using:', orient, 'orientations', pix_per_cell,
		  'pixels per cell and', cell_per_block, 'cells per block')
	print('Feature vector length:', len(X_train[0]))
	# Use a linear SVC
	svc = LinearSVC()
	# svc = SVC(C=1.0, probability=True)
	# Check the training time for the SVC
	t = time.time()
	svc.fit(X_train, y_train)
	t2 = time.time()
	print(round(t2 - t, 2), 'Seconds to train SVC...')
	# Check the score of the SVC
	print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
	# Check the prediction time for a single sample
	t = time.time()
	filename = "svc.pkl"
	joblib.dump(svc, filename)
	pickle.dump(X_scaler,open("scaler.pkl","wb"))
```

Notes:

* the lenght of the features vector is 6108
* in order to train the model, please execute the python program with the `--train` parameter (`python main.py --train`)
* training of a LinearSVC model is very fast, but it generates many false positives during the video processing
* training of a SVC model that returns probability estimates (sklearn.svm.SVC) can greatly reduce the false positives, but both training time and predictions are very slow. I therefore decided to keep using the LinearSVC model and use other methods to reject the false positives
* the SVC model and the scaler function are persisted to disk, to be used for the prediction stage (`python main.py --video` or `python main.py --images`)

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I actually have two versions of this functionality in my code, but I only use the one that provides better performances.

##### Version 1

The first version, provided since the beginning of the Udacity course, is available in the `slide_window` function:

```python
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
				 xy_window=(64, 64), xy_overlap=(0.5, 0.5),window_list=None):
```

This version applies the HOG features extraction for each window during the sliding and, therefore, it is a bit inefficient

##### Version 2

This version, provided at a later stage in the Udacity course, is more efficient. The sliding window is implemented starting from line 675 in main.py:

```python
for xb in range(nxsteps):
		for yb in range(nysteps):
			ypos = yb*cells_per_step
			xpos = xb*cells_per_step
```

I used different scales to resize the image:

```python
hot_windows = detect_vehicles(image, 0.9)
hot_windows.extend(detect_vehicles(image, 1.0))
hot_windows.extend(detect_vehicles(image, 1.3))
hot_windows.extend(detect_vehicles(image, 1.7))
hot_windows.extend(detect_vehicles(image, 2.0))
hot_windows.extend(detect_vehicles(image, 2.3))
hot_windows.extend(detect_vehicles(image, 2.5))
```



####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Before applying my pipeline to the videos, I tried it on the test images. Here are the resulting images:

| Original | Hot windows | Heatmap | Labeled boxes | Final |
| ------------- |:-------------:| :-------------:| :-------------:| :-------------:| 
| ![t1][t1] | ![w1][w1] |![h1][h1] | ![l1][l1] | ![f1][f1] |
| ![t2][t2] | ![w2][w2] |![h2][h2] | ![l2][l2] | ![f2][f2] |
| ![t3][t3] | ![w3][w3] |![h3][h3] | ![l3][l3] | ![f3][f3] |
| ![t4][t4] | ![w4][w4] |![h4][h4] | ![l4][l4] | ![f4][f4] |
| ![t5][t5] | ![w5][w5] |![h5][h5] | ![l5][l5] | ![f5][f5] |
| ![t6][t6] | ![w6][w6] |![h6][h6] | ![l6][l6] | ![f6][f6] |
--

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Project's video result is available at this [link](./out_project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I used the approach suggested during the Udacity course and I added something very naive to keep track of the heatmap over time:

* I recorded the positions of positive detections in each frame of the video
* From the positive detections I created a (global) heatmap and then thresholded that map to identify vehicle positions
* I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap
* I then assumed each blob corresponded to a vehicle
* I constructed bounding boxes to cover the area of each blob detected
* I applied a very simple transformation to the global heatmap, so that values coming from previous frame would become less important in the detection of the bounding box `heat = apply_threshold(heatmap/2, 6)`
* I re-iterated the same process for each frame of the video and used the previous heatmap as a base to calculate the current heatmap

The reason why I did so, is that I wanted to eliminate false positives that only appeared for 1 or 2 frames in the video processing: the heatmap keep tracks of the position of the vehicles in the previous frames and sums up to the position in the current frames. The values in the heatmap are divided by two and thresholded at each step to make sure that old values slowly fade away and become less and less important for the bounding box detection.

Code:

```python
def process_single(image):
	
	global heat

	
	draw_img = np.copy(image)
	hot_windows = detect_vehicles(image, 1.0)
	hot_windows.extend(detect_vehicles(image, 1.5))
	hot_windows.extend(detect_vehicles(image, 2.0))
	hot_windows.extend(detect_vehicles(image, 2.5)) 
	
	heatmap = add_heat(heatmap=heat, bbox_list=hot_windows)
	heatmap = apply_threshold(heatmap, 2)
	
	labels = label(heat)
	window_img = draw_boxes(draw_img, hot_windows, color=(255, 138, 0), thick=1)
	window_img = draw_labeled_bboxes(draw_img, labels, color=(12, 138, 255), thick=2)
	
	# keep track of previous heatmaps, in order to be able to eliminate false positives
	heat = apply_threshold(heatmap/2, 6)
	
	return window_img
```

Some test videos are available:

* [original test video](./test_video.mp4)
* [hot windows](./videos/bboxes.mp4)
* [heatmap](./videos/bboxes.mp4)
* [final](./videos/out_test_video.mp4)

#### Bonus

* Advanced line finding and vehicle detection combined [video](./combined.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Some considerations:

* overall performances are not good enough to be used to process a video stream in real-time. This is mainly due to the fact that windows are processed one after the other: it would be beneficial to use a concurrent approach where multiple threads scans different regions of the image
* false positives and false negatives detection must be improved, because it can cause great danger to detect a car that is not there or, even worse probably, to not detect a car
* the HOG features extraction depends on many parameters. Those parameters work quite well with the project's video, but it unlikely they can work with other videos recorded under different conditions
* I would like to improve the suppression of redundant bounding boxes using a non-maximum suppression approach, as described at http://www.computervisionblog.com/2011/08/blazing-fast-nmsm-from-exemplar-svm.html
* I had many troubles in finding a trade-off between the number of windows to use in the sliding window phase, the accuracy of vehicles detection, and the performances (about 1 FPS). I would like to find if there are any good approaches that allow to solve similar problems in real-time



