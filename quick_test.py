#!python
"""
Using Python 2.7 and Caffe to classify images in `bearSightings` dataset.
"""

import fnmatch
import numpy as np 
import os 
import sys

import caffe 

# Set root to caffe
caffe_root = '/opt/caffe'
# Set the path to model file and pretrained model weights
MODEL_FILE = os.path.join(caffe_root, 'models/bvlc_reference_caffenet/deploy.prototxt')
PRETRAINED = os.path.join(caffe_root, 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')

def gen_find(topdir, pattern):
	""" Generator which yields files in a directory matching a pattern """
	for path, dirlist, filelist in os.walk(topdir):
		for name in fnmatch.filter(filelist, pattern):
			yield os.path.join(path, name)

# Load the network
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
					   mean=np.load(os.path.join(caffe_root, 
					   			    'python/caffe/imagenet/ilsvrc_2012_mean.npy')),
					   channel_swap=(2,1,0),
					   raw_scale=255,
					   image_dims=(256, 256))

# Set the phase to test
net.set_phase_test()

# Set the mode to GPU
net.set_mode_gpu()

# Get the filenames of the images 
image_paths = list(gen_find('./data/bearSightings', 'bear*.jpg'))