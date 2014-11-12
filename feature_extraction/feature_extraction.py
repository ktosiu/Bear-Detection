#!python
"""
Example of how to perform feature extraction.

As far as I understand, you cannot avoid the computational inefficiency of 
passing through the entire network without making some weird changes.

Also inefficient is the fact that, to run this as a script, you'd be loading 
the network each time.
"""

import os 
import sys
import numpy as np 

import caffe

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
caffe_input = np.asarray([net.preprocess(net.inputs[0], resized)])

# Forward the result (note that we can't seem to use just `net.forward`)
out = net.forward_all(**{net.inputs[0], caffe_input})

# Extract the features from a layer
layer_name = 'fc8'
feat = net.blobs[layer_name]

# Flatten the features, result is a vector
feat = feat.flatten() 
np.savetxt('feat.txt', feat)



