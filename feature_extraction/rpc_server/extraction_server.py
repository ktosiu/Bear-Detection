#!python2.7
"""
A server which runs Caffe and can be called using XMLRPC in order to perform
feature extraction.
"""

from __future__ import print_function

from SimpleXMLRPCServer import SimpleXMLRPCServer
from SimpleXMLRPCServer import SimpleXMLRPCRequestHandler

import argparse
import numpy as np
import os
import sys
import tempfile
import xmlrpclib

from scipy.ndimage import imread

# Try importing Caffe as is, if not, assume it's in /opt/caffe/python
try:
    import caffe
except ImportError as e:
    sys.path.append('/opt/caffe/python')
    import caffe


class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ('/RPC2',)

def upload_extract(img, layer_name=None):
    """
    Take an image and extract features from it.

    Parameters
    ----------
    img : xmlrpclib.Binary
        The binary serialized image from which to extract the features
    layer_name : str 
        The name of the layer whose output to use as features.
        If unspecified, use the classification probabilities.
    """
    
    if layer_name == None:
        layer_name = list(net.blobs.keys())[-1]

    # Create a temporary file to store the image
    tmp_fd, tmp_path = tempfile.mkstemp(prefix='image~')
    with open(tmp_path, 'wb') as f:
        f.write(img.data)
    os.close(tmp_fd)

    # Load the image
    input_image = caffe.io.load_image(tmp_path)

    # Process the image
    scores = net.predict([input_image])

    # Extract the features
    feat = net.blobs[layer_name].data
    feat = feat.flatten()

    # Convert features to a list and return the results
    ret  = feat.tolist()
    return ret 

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=8888)
    parser.add_argument("--gpu", action='store_true')

    # Get arguments
    args = parser.parse_args()

    # Would perform argument validation here

    print("Setting up Caffe")
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
    if args.gpu:
        net.set_mode_gpu()
    else:
        net.set_mode_cpu()
    net.set_mean('data', np.load(os.path.join(caffe_root,'python/caffe/imagenet/ilsvrc_2012_mean.npy'))) 
    net.set_raw_scale('data', 255)  
    net.set_channel_swap('data', (2,1,0))

    # Make the net available to other functions
    global net 

    # Set up the server
    server = SimpleXMLRPCServer((args.host, args.port),
                                requestHandler=RequestHandler,
                                allow_none=True)
    server.register_introspection_functions()
    server.register_function(upload_extract)

    # Print information once setup is complete
    print("Server set up at:", args.host)
    print("Listening on port:", args.port)
    print("Using GPU:", args.gpu)

    # Actually start the server
    server.serve_forever()

if __name__ == "__main__":
    main(sys.argv)