#!python2.7
"""
A server which runs Caffe and can be called using XMLRPC in order to perform
feature extraction.
"""

from __future__ import print_function

from SimpleXMLRPCServer import SimpleXMLRPCServer
from SimpleXMLRPCServer import SimpleXMLRPCRequestHandler

import numpy as np
import os
import sys

# Try importing Caffe as is, if not, assume it's in /opt/caffe/python
try:
    import caffe
except ImportError as e:
    sys.path.append('/opt/caffe/python')
    import caffe


# Default Caffe setup (may want to move this into a function)
caffe_root = '/opt/caffe'
# set path to model file and pretrained model weights
MODEL_FILE = os.path.join(caffe_root, 'models/bvlc_reference_caffenet/deploy.prototxt')
PRETRAINED = os.path.join(caffe_root, 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')
# load the network
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       mean = np.load(os.path.join(caffe_root, 'python/caffe/imagenet/ilsvrc_2012_mean.npy')),
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))

# Set the phase to test
net.set_phase_test()
# Set the mode
net.set_mode_cpu()


def extract_features(img_path, layer_name=None, window=None, gpu=False):
    """
    Perform feature extraction using Caffe on the image located at
    `img_path`, returning the result as a list of floats.

    Specifying the layer name is optional, as is specifying
    the particular window over which to extract.

    If the `gpu` flag is set to `True`, then attempt to use Caffe's GPU
    acceleration when extracting features.
    """
    # Set up network for extraction
    if gpu:
        net.set_mode_gpu()
    # If layer_name unspecified, use second from last layer
    if layer_name == None:
        layer_name = list(net.blobs.keys())[-2]

    # Load the image from the path
    try:
        img_path    = os.path.expanduser(img_path)
        img_path    = os.path.abspath(img_path)
        input_image = caffe.io.load_image(img_path)
    except Exception as e:
        raise(e)
    # Crop and resize the image
    cropped = input_image # TODO
    resized = caffe.io.resize_image(cropped, net.image_dims)

    # Preprocess for Caffe
    caffe_input = np.asarray([net.preprocess(net.inputs[0], resized)])

    # Forward pass through Caffe network
    out = net.forward_all(**{net.inputs[0]: caffe_input})

    # Extract the features, convert to a list, return them
    feat = net.blobs[layer_name].data
    feat = feat.flatten()

    ret  = feat.tolist()
    return ret

def upload_image(img, img_name):
    """
    Upload an image, save it in the local directory under the supplied name.

    Parameters:
    img : xmlrpclib.Binary 
        An image in xmlrpclib's binary format
    img_name : str 
        The name to give to the image.
    """
    name = os.path.basename(img_name)
    with open(name, 'wb') as f:
        f.write(img.data)


# Setup for the server
port_num = 8888
hostname = "0.0.0.0"
# restrict to a particular path 
class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ('/RPC2',)

# Create server 
server = SimpleXMLRPCServer((hostname, port_num), 
                            requestHandler=RequestHandler,
                            allow_none=True)
print("Listening on port:", port_num)
server.register_introspection_functions()

# Register functions
server.register_function(extract_features)

# Start main loop
server.serve_forever()