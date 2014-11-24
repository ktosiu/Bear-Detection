#!python3
"""
Use Caffe to perform feature extraction on an entire directory of images.
"""

from __future__ import print_function

import argparse
import glob
import os 
import sys
import numpy as np 

try:
    import caffe
except ImportError as e:
    sys.path.append('/opt/caffe/python')
    import caffe

def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "pathname",
        help="unix-style pathname pattern to get images from")

    parser.add_argument(
        "output_dir",
        help="The output directory in which to store the extracted features",
        default="./feat/")

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

    input_files     = glob.iglob(args.pathname)
    output_dir      = args.output_dir
    layer_name      = args.layer_name
    
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

    for path in input_files:
        # Load the image
        try:
            print(path)
            output_name = os.path.splitext(os.path.basename(path))[0]
            output_path = os.path.join(output_dir, output_name) + '.' + layer_name 
            print(output_path)
            input_image = caffe.io.load_image(path) 
        except Exception as e:
            raise(e)

        # Resize the image for the net
        resized = caffe.io.resize_image(input_image, net.image_dims)

        # Preprocess for caffe
        caffe_input = np.asarray([net.preprocess(net.inputs[0], resized)])

        # Forward the result (note that we can't seem to use just `net.forward`)
        out = net.forward_all(**{net.inputs[0]: caffe_input})

        # Extract the features from the layer
        feat = net.blobs[layer_name].data
        # Flatten the features to convert to a vector
        feat = feat.flatten()

        # Save the result
        np.savetxt(output_path, feat)

if __name__ == "__main__":
    main(sys.argv)
