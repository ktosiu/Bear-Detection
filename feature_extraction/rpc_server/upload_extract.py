#python2.7
"""
A wrapper script which allows for command line uploading of images and feature
extraction, with Caffe running on a remote machine.

Usage:
    $ python upload_extract.py <remote_host> <img_path> [--layer <name>]

The `remote_host` is generally of the form http://<ip_addr>:<port_num>

Example:
    $ python upload_extract.py http://54.164.113.154:8888 ../bear_0041.jpg --layer fc8
"""

from __future__ import print_function

import argparse
import numpy as np 
import os 
import sys 
import xmlrpclib

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("remote_host", 
                        help="IP address and port of remote machine to connect to.")
    parser.add_argument("img_path", 
                        help="Path to image to upload and extract features from.")
    parser.add_argument("--layer",
                        help="layer from which to extract features",
                        default=None)
    parser.add_argument("-p",
                        help="print to stdout instead of saving the output",
                        action="store_true")

    # Get argument values
    args = parser.parse_args()
    remote_host = args.remote_host
    img_path    = args.img_path
    layer_name  = args.layer

    # Set up server proxy
    proxy = xmlrpclib.ServerProxy(args.remote_host, allow_none=True)

    # Load image for upload
    img_file = xmlrpclib.Binary(open(args.img_path, 'rb').read())

    # Upload the image and get the features
    response = proxy.upload_extract(img_file, args.layer)

    # Save the response (or print it, depending on the arguments)
    if args.p:
        print("\n".join([str(x) for x in response]))
    else:
        base = os.path.basename(args.img_path)
        output_file = base + '.txt'
        np.savetxt(output_file, response)


if __name__ == "__main__":
    main(sys.argv)