#!python3
"""
A script to extract features.

Example:
    $ python extract_script.py hello.jpg --gpu --window 0 100 100 200 --output out.txt

A more sophisticated script might be able to do this in bulk, to avoid the 
overhead of initializing Caffe each time the script is to be run, or be able 
to run feature extraction for multiple windows on the same image.
"""

from __future__ import print_function

import argparse
import os 
import sys
import numpy as np 

import caffe


def winfunc(s):
    """ Convert a string to a list of integers for an image window"""
    try: 
        winlst = list(map(int, s.split(',')))
        return winlst
    except:
        raise TypeError("Invalid string specifying window:%s"%s)


def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file",
        help="input the name of an image file to extract features from")

    parser.add_argument(
        "--window",
        help="the window within the image to crop",
        default="-1,-1,-1,-1",
        type=winfunc)

    parser.add_argument(
        "--output_file",
        help="output filename, default is 'feat.txt'",
        default=os.path.abspath(os.path.join(os.path.curdir, 'feat.txt')))

    parser.add_argument(
        "--layer_name",
        help="name of layer from which features are to be extracted",
        default=None)

    parser.add_argument(
        "--gpu",
        action='store_true',
        help='switch to use GPU processing')

    args = parser.parse_args()

    print(args) # REMOVE

    input_file  = args.input_file
    window      = args.window
    layer_name  = args.layer_name
    output_file = args.output_file

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
    # Set the mode 
    if args.gpu:
        net.set_mode_gpu()
        print("GPU mode")
    else:
        net.set_mode_cpu()
        print("CPU mode")

    # If the layer is unspecified, use the penultimate layer
    if layer_name == None:
        layer_name = list(net.blobs.keys())[-2]
        print("Extracting features from layer:",layer_name)

    # Load the image
    try:
        input_image = caffe.io.load_image(input_file) 
    except Exception as e:
        raise(e)

    # Extract the window from the image, if no window is specified, don't crop
    if all([i == -1 for i in window]):
        cropped_image = input_image
    else:
        try:
            cropped_image = input_image[window[0]:window[2], window[1]:window[2]]
        except Exception as e:
            raise(e)

    # Resize the image for the net
    resized = caffe.io.resize_image(cropped_image, net.image_dims)

    # Preprocess for caffe
    caffe_input = np.asarray([net.preprocess(net.inputs[0], resized)])

    # Forward the result (note that we can't seem to use just `net.forward`)
    out = net.forward_all(**{net.inputs[0]: caffe_input})

    # Extract the features from the layer
    feat = net.blobs[layer_name].data
    # Flatten the features to convert to a vector
    feat = feat.flatten()

    # Save the result
    np.savetxt(output_file, feat)

if __name__ == "__main__":
    main(sys.argv)
















