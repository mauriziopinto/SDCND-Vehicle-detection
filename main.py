# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images

import glob
import time
import sys
import pickle

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from cv2 import HOGDescriptor
from skimage.feature import hog
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#from sklearn.svm import LinearSVC
#from sklearn.svm import SVC
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip

from sklearn.neural_network import MLPClassifier

def get_hog_features(img, orient, pix_per_cell, cell_per_block,
					 vis=False, feature_vec=True):
	"""
	Define a function to return HOG features and visualization

	Args:
		img
		orient
		pix_per_cell
		cell_per_block
		vis
		feature_vec
 
	Returns:
		the HOG features extracted from the image
	"""
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


def bin_spatial(img, size=(32, 32)):
	"""
	Define a function to compute binned color features

	Args:
		img
		size

	Returns:
		the binned color features
	"""
	# Use cv2.resize().ravel() to create the feature vector
	#features = cv2.resize(img, size).ravel()

	color1 = cv2.resize(img[:,:,0], size).ravel()
	color2 = cv2.resize(img[:,:,1], size).ravel()
	color3 = cv2.resize(img[:,:,2], size).ravel()
	# Return the feature vector
	return np.hstack((color1, color2, color3)) #features


def color_hist(img, nbins=32): #, bins_range=(0, 256)
	"""
	Define a function to compute color histogram features

	Args:
		img
		nbins

	Returns:
		the channels' histograms
	"""
	# Compute the histogram of the color channels separately
	channel1_hist = np.histogram(img[:, :, 0], bins=nbins)
	channel2_hist = np.histogram(img[:, :, 1], bins=nbins)
	channel3_hist = np.histogram(img[:, :, 2], bins=nbins)
	# Concatenate the histograms into a single feature vector
	hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
	# Return the channels' histograms
	return hist_features


def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
					 hist_bins=32, orient=9,
					 pix_per_cell=8, cell_per_block=2, hog_channel=0,
					 spatial_feat=True, hist_feat=True, hog_feat=True):
	"""
	Define a function to extract features from a list of images
	Have this function call bin_spatial() and color_hist()

	Args:
		imgs
		color_space
		spatial_size
		hist_bins
		orient
		pix_per_cell
		cell_per_block
		hog_channel
		spatial_feat
		hist_feat
		hog_feat

	Returns:
		the list of feature vectors
	"""
	# Create a list to append feature vectors to
	features = []
	# Iterate through the list of images
	for file in imgs:
		file_features = []
		# Read in each one by one
		image = mpimg.imread(file)
		# apply color conversion if other than 'RGB'
		if color_space != 'RGB':
			if color_space == 'HSV':
				feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
			elif color_space == 'LUV':
				feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
			elif color_space == 'HLS':
				feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
			elif color_space == 'YUV':
				feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
			elif color_space == 'YCrCb':
				feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
		else:
			feature_image = np.copy(image)

		if spatial_feat == True:
			spatial_features = bin_spatial(feature_image, size=spatial_size)
			file_features.append(spatial_features)
		if hist_feat == True:
			# Apply color_hist()
			hist_features = color_hist(feature_image, nbins=hist_bins)
			file_features.append(hist_features)
		if hog_feat == True:
			# Call get_hog_features() with vis=False, feature_vec=True
			if hog_channel == 'ALL':
				hog_features = []
				for channel in range(feature_image.shape[2]):
					hog_features.append(get_hog_features(feature_image[:, :, channel],
														 orient, pix_per_cell, cell_per_block,
														 vis=False, feature_vec=True))
				hog_features = np.ravel(hog_features)
			else:
				hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
												pix_per_cell, cell_per_block, vis=False, feature_vec=True)
			# Append the new feature vector to the features list
			file_features.append(hog_features)
		features.append(np.concatenate(file_features))
	# Return list of feature vectors
	return features


def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
				 xy_window=(64, 64), xy_overlap=(0.5, 0.5),window_list=None):
	"""
	Define a function that takes an image,
	start and stop positions in both x and y,
	window size (x and y dimensions),
	and overlap fraction (for both x and y)

	Args:
		img
		x_start_stop
		y_start_stop
		xy_window
		xy_overlap
		window_list

	Returns:
		the list of windows
	"""
	# If x and/or y start/stop positions not defined, set to image size
	if x_start_stop[0] == None:
		x_start_stop[0] = 0
	if x_start_stop[1] == None:
		x_start_stop[1] = img.shape[1]
	if y_start_stop[0] == None:
		y_start_stop[0] = 0
	if y_start_stop[1] == None:
		y_start_stop[1] = img.shape[0]
	# Compute the span of the region to be searched
	xspan = x_start_stop[1] - x_start_stop[0]
	yspan = y_start_stop[1] - y_start_stop[0]
	# Compute the number of pixels per step in x/y
	nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
	ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
	# Compute the number of windows in x/y
	nx_windows = np.int(xspan / nx_pix_per_step) - 1
	ny_windows = np.int(yspan / ny_pix_per_step) - 1
	# Initialize a list to append window positions to
	if window_list == None:
		window_list = []
	# Loop through finding x and y window positions
	# Note: you could vectorize this step, but in practice
	# you'll be considering windows one by one with your
	# classifier, so looping makes sense
	for ys in range(ny_windows):
		for xs in range(nx_windows):
			# Calculate window position
			startx = xs * nx_pix_per_step + x_start_stop[0]
			endx = startx + xy_window[0]
			starty = ys * ny_pix_per_step + y_start_stop[0]
			endy = starty + xy_window[1]

			# Append window position to list
			window_list.append(((startx, starty), (endx, endy)))
	# Return the list of windows
	return window_list


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
	"""
	Define a function to draw bounding boxes

	Args:
		img
		bboxes
		color
		thick

	Returns:
		the image copy with boxes drawn
	"""
	# Make a copy of the image
	imcopy = np.copy(img)
	# Iterate through the bounding boxes
	for bbox in bboxes:
		# Draw a rectangle given bbox coordinates
		cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
	# Return the image copy with boxes drawn
	return imcopy

def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
						hist_bins=32, orient=9,
						pix_per_cell=8, cell_per_block=2, hog_channel=0,
						spatial_feat=True, hist_feat=True, hog_feat=True):
	"""
	Apply pipeline to a single image

	Args:
		img
		color_space
		spatial_size
		hist_bins
		orient
		pix_per_cell
		cell_per_block
		hog_channel
		spatial_feat
		hist_feat
		hog_feat
	Returns:
		concatenated array of features
	"""
	# 1) Define an empty list to receive features
	img_features = []
	# 2) Apply color conversion if other than 'RGB'
	if color_space != 'RGB':
		if color_space == 'HSV':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
		elif color_space == 'LUV':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
		elif color_space == 'HLS':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
		elif color_space == 'YUV':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
		elif color_space == 'YCrCb':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
	else:
		feature_image = np.copy(img)
	# 3) Compute spatial features if flag is set
	if spatial_feat == True:
		spatial_features = bin_spatial(feature_image, size=spatial_size)
		# 4) Append features to list
		img_features.append(spatial_features)
	# 5) Compute histogram features if flag is set
	if hist_feat == True:
		hist_features = color_hist(feature_image, nbins=hist_bins)
		# 6) Append features to list
		img_features.append(hist_features)
	# 7) Compute HOG features if flag is set
	if hog_feat == True:
		if hog_channel == 'ALL':
			hog_features = []
			for channel in range(feature_image.shape[2]):
				hog_features.append(get_hog_features(feature_image[:, :, channel],
													 orient, pix_per_cell, cell_per_block,
													 vis=False, feature_vec=True))
			hog_features = np.concatenate(hog_features)
		else:
			hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
											pix_per_cell, cell_per_block, vis=False, feature_vec=True)
		# 8) Append features to list
		img_features.append(hog_features)

	# 9) Return concatenated array of features
	return np.concatenate(img_features)


def search_windows(img, windows, clf, scaler, color_space='RGB',
				   spatial_size=(32, 32), hist_bins=32,
				   hist_range=(0, 256), orient=9,
				   pix_per_cell=8, cell_per_block=2,
				   hog_channel=0, spatial_feat=True,
				   hist_feat=True, hog_feat=True):
	"""
	Define a function you will pass an image
	and the list of windows to be searched (output of slide_windows())

	Args:
		img
		windows
		clf
		scaler
		color_space
		spatial_size
		hist_bins
		hist_range
		orient
		pix_per_cell
		cell_per_block
		hog_channel
		spatial_feat
		hist_feat
		hog_feat
	Returns:
		list of windows where a car has been detected
	"""
	t1 = time.clock()
	# 1) Create an empty list to receive positive detection windows
	on_windows = []
	# 2) Iterate over all windows in the list
	for window in windows:
		# 3) Extract the test window from original image
		test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
		# 4) Extract features for that window using single_img_features()
		features = single_img_features(test_img, color_space=color_space,
									   spatial_size=spatial_size, hist_bins=hist_bins,
									   orient=orient, pix_per_cell=pix_per_cell,
									   cell_per_block=cell_per_block,
									   hog_channel=hog_channel, spatial_feat=spatial_feat,
									   hist_feat=hist_feat, hog_feat=hog_feat)
		
		# 5) Scale extracted features to be fed to classifier
		test_features = scaler.transform(np.array(features).reshape(1, -1))
		# 6) Predict using your classifier
		prediction = clf.decision_function(test_features) #clf.predict(test_features)
		# 7) If positive (prediction == 1) then save the window
		if prediction > 0.65: #prediction == 1: # #
			on_windows.append(window)
	# 8) Return windows for positive detections
	#print(time.clock() - t1)
	return on_windows



def train():
	"""
	Train a model and save the model and the scaler on disk

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
	#svc = LinearSVC()
	mlp = MLPClassifier()
	# svc = SVC(C=1.0, probability=True)
	# Check the training time for the SVC
	t = time.time()
	mlp.fit(X_train, y_train)
	t2 = time.time()
	print(round(t2 - t, 2), 'Seconds to train model...')
	# Check the score of the model
	print('Test Accuracy of MLP model = ', round(mlp.score(X_test, y_test), 4))
	# Check the prediction time for a single sample
	t = time.time()
	filename = "mlp.pkl"
	joblib.dump(mlp, filename)
	pickle.dump(X_scaler,open("scaler.pkl","wb"))


def add_heat(heatmap, bbox_list):
	"""
	Update the heatmap

	Args:
		heatmap
		bbox_list
		
	Returns:
		the updated heatmap
	"""
	# Iterate through list of bboxes
	for box in bbox_list:
		# Add += 1 for all pixels inside each bbox
		# Assuming each "box" takes the form ((x1, y1), (x2, y2))
		heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

	# Return updated heatmap
	return heatmap


def apply_threshold(heatmap, threshold):
	"""
	Apply a threshold (remove pixels below a threshold) to a heatmap

	Args:
		heatmap
		threshold
		
	Returns:
		the heatmap after the thresholding
		
	"""
	# Zero out pixels below the threshold
 
	heatmap[heatmap <= threshold] = 0
	# Return thresholded map
	return heatmap


def draw_labeled_bboxes(img, labels, color, thick):
	"""
	Draw labeled bounding boxes

	Args:
		img
		labels
		color
		thick
		
	Returns:
		the image with the labeled bounding boxes
		
	"""
	# Iterate through all detected cars
	for car_number in range(1, labels[1]+1):
		# Find pixels with each car_number label value
		nonzero = (labels[0] == car_number).nonzero()
		# Identify x and y values of those pixels
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		# Define a bounding box based on min/max x and y
		bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
		# Draw the box on the image
		cv2.rectangle(img, bbox[0], bbox[1], color, thick)
	# Return the image
	return img



def gethotwindows(image):
	"""
	Select the regions in an image where a car has been detected

	Args:
		image
		
	Returns:
		the list of windows where a car has been detected
		all the windows that have been analyzed
		
	"""
	hot_windows = []
	windows = []

	y_start_stop = [int(image.shape[0]/2), int(image.shape[0]/2)+120] 
	windows.extend(slide_window(image, x_start_stop=[int(image.shape[1]*0.20), int(image.shape[1]*0.80)], y_start_stop=y_start_stop, xy_window=(64,64), xy_overlap=(0.5, 0.5)))

	y_start_stop = [int(image.shape[0]/2)+32, int(image.shape[0]/2)+160] 
	windows.extend(slide_window(image, x_start_stop=[int(image.shape[1]*0.10), int(image.shape[1])], y_start_stop=y_start_stop, xy_window=(96,96), xy_overlap=(0.7, 0.7)))

	y_start_stop = [int(image.shape[0]/2)+170, image.shape[0]] 
	windows.extend(slide_window(image, x_start_stop=[0, image.shape[1]], y_start_stop=y_start_stop, xy_window=(128, 128), xy_overlap=(0.6, 0.6)))

	
	#print(len(windows))
		
	hot_windows = search_windows(image, windows, svm, X_scaler, color_space=color_space,
								 spatial_size=spatial_size, hist_bins=hist_bins,
								 orient=orient, pix_per_cell=pix_per_cell,
								 cell_per_block=cell_per_block,
								 hog_channel=hog_channel, spatial_feat=spatial_feat,
								 hist_feat=hist_feat, hog_feat=hog_feat)

	return hot_windows, windows


def process_single(image):
	"""
	Ápply the full pipeline to a single image

	Args:
		image
		
	Returns:
		the image with the detected vehicles
		
	"""
	

	global heat

	
	draw_img = np.copy(image)
	hot_windows = detect_vehicles(image, 1.0)
	hot_windows.extend(detect_vehicles(image, 1.5))
	hot_windows.extend(detect_vehicles(image, 2.0))
	hot_windows.extend(detect_vehicles(image, 2.5))
	
	#heat = np.zeros_like(image[:, :, 0]).astype(np.float)
	
	heatmap = add_heat(heatmap=heat, bbox_list=hot_windows)
	heatmap = apply_threshold(heatmap, 2)
	
	labels = label(heat)
	#window_img = draw_boxes(draw_img, hot_windows, color=(255, 138, 0), thick=1)
	window_img = draw_labeled_bboxes(draw_img, labels, color=(12, 138, 255), thick=2)
	
	# keep track of previous heatmaps, in order to be able to eliminate false positives
	heat = apply_threshold(heatmap/1.5, 1)
	
	return window_img

def process_video(output_path, input_path):
	"""
	Process a video

	Args:
		input_path
		output_path
		
	Returns:
		the processed video
		
	"""
	
	input_file = VideoFileClip(input_path)
	standard_clip = input_file.fl_image(process_single) #NOTE: this function expects color images!!
	standard_clip.write_videofile(output_path, audio=False, threads=4)
	return output_path


def convert_color(img, conv='RGB2YCrCb'):
	"""
	Convert from a color space to a different color space. Needed by detect_vehicles

	Args:
		img
		conv
		
	Returns:
		the image after the color space conversion
		
	"""
	
	if conv == 'RGB2YCrCb':
		return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
	if conv == 'BGR2YCrCb':
		return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
	if conv == 'RGB2LUV':
		return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

def detect_vehicles(img, scale):
	"""
	Optimized pipeline on a single image

	Args:
		img
		scale
		
	Returns:
		the list of windows where cars have been detected
	"""
	
	
	draw_img = np.copy(img)
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
	
	hot_windows = []

	for xb in range(nxsteps):
		for yb in range(nysteps):
			ypos = yb*cells_per_step
			xpos = xb*cells_per_step
			# Extract HOG for this patch
			hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
			hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
			hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
			hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

			xleft = xpos*pix_per_cell
			ytop = ypos*pix_per_cell

			# Extract the image patch
			subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
		  
			# Get color features
			spatial_features = bin_spatial(subimg, size=spatial_size)
			hist_features = color_hist(subimg, nbins=hist_bins)

			# Scale features and make a prediction
			test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
			#test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
			test_prediction = svm.predict(test_features)
			
			if test_prediction == 1:
				xbox_left = np.int(xleft*scale)
				ytop_draw = np.int(ytop*scale)
				win_draw = np.int(window*scale)
				#cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
				hot_windows.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))

	return hot_windows


### Tweak these parameters and see how the results change.
color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32)  # Spatial binning dimensions
hist_bins = 32  # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off
ystart = 400 # y start of the region to process
ystop = 656 # y stop of the region to process

# global heatmap
heat = np.zeros((720,1280,3)).astype(np.float)

if __name__ == "__main__":

	arg = sys.argv[1:]

	if arg:
		mode = arg[0]
	else:
		mode = "--help"

	if mode == "--train":
		print("Training mode selected.")
		print("Running...")
		train()
		print("Completed.")
	elif mode == "--images":
		print("Test images mode selected.")
		print("Running...")
		X_scaler = pickle.load(open("scaler.pkl", "rb"))
		svm = joblib.load("mlp.pkl")
		images = glob.glob("test_images/*")
		for file in images:
			print("Processing", file)
			image = mpimg.imread(file)
			plt.imsave("out_" + file + ".png", process_single(image))
		print("Completed.")
	elif mode == "--video":
		print("Video mode selected")
		print("Running...")
		X_scaler = pickle.load(open("scaler.pkl", "rb"))
		svm = joblib.load("mlp.pkl")
		#output = process_video('./test_out.mp4', './test_video.mp4')
		#1#output = process_video('./out_project_video.mp4', './project_video.mp4')
		output = process_video('./combined.mp4', './carnd_p4_output.mp4')

		print("Completed.")
	elif mode == "--help":
		print("\n---------------------------------\nUdacity SDCND - Vehicle detection\n---------------------------------\n")
		print(" - --train : trains a neural network model and saves model and scaler to disk")
		print(" - --images : executes vehicle detection on the images in the test_images folder")
		print(" - --video : executed vehicle detection on the project_video.mp4 file")
		print(" - --help : display this help\n\n")
	else:
		print("\n---------------------------------\nUdacity SDCND - Vehicle detection\n---------------------------------\n - Valid options are --train --images --video.\n - Use python main.py --help for more details")
	