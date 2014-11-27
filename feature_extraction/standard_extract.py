#!python2.7
"""
Perform feature extraction in the same way that Caffe expects the images to be 
set up, which entails a certain amount of cropping and mirroring, rather than 
just sending the image straight through.
"""

import os 
import sys
import numpy as np 

# Specify root directory for caffe
caffe_root = '/opt/caffe'
# Set the path to model file and pretrained model weights
MODEL_FILE = os.path.join(caffe_root, 'models/bvlc_reference_caffenet/deploy.prototxt')
PRETRAINED = os.path.join(caffe_root, 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')

# Load the network
net = caffe.Classifier(MODEL_FILE, PRETRAINED, 
                       mean=np.load(os.path.join(caffe_root, 'python/caffe/imagenet/ilsvrc_2012_mean.npy')), 
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))

# Set the phase to test
net.set_phase_test()
# Set the mode to GPU
net.set_mode_gpu()

# An image file for testing
IMAGE_FILE = os.path.join(caffe_root, 'examples/images/cat.jpg')

# Load the image
input_image = caffe.io.load_image(IMAGE_FILE)

# Specify a window (x1, y1, x2, y2)
window = [100, 100, 300, 400]

# Crop the image
cropped_image = input_image[window[0]:window[2], window[1]:window[3]]

# Resize the image for the net
resized = caffe.io.resize_image(cropped_image, net.image_dims)

# Preprocess for caffe
caffe_input = np.asarray(net.preprocess(net.inputs[0], resized))
