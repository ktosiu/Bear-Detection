#!python2.7
"""
A wrapper script which allows for command line feature extraction using Caffe,
with Caffe on a remote machine.

Assuming Caffe is running on a remote machine that you have access to, on your
local machine you would run this script via

    $ python rpc_wrapper.py <...args...>

Example:
    $ python rpc_wrapper.py "http://54.173.163.181:8888" "~/Bear-Detection/bear_0041.jpg"

Note:
    * The `input_file` argument is the path to the input file on the REMOTE machine
    * The script saves the resulting features in the local directory.
"""

from __future__ import print_function

import argparse
import numpy as np
import os
import sys
import xmlrpclib


def winfunc(s):
    """ Convert a string to a list of integers for an image window"""
    try: 
        winlst = list(map(int, s.split(',')))
        return winlst
    except:
        raise TypeError("Invalid string specifying window:%s"%s)

def main(argv):

    # Parse the arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "remote_host",
        help="IP address and port of remote machine on which Caffe is running"
             "For example, `http://54.173.163.181:8888")

    parser.add_argument(
        "input_file",
        help="input the name of an image file to extract features from")

    parser.add_argument(
        "--window",
        help="the window within the image to crop",
        default="-1,-1,-1,-1",
        type=winfunc)

    parser.add_argument(
        "--layer",
        help="name of layer from which features are to be extracted",
        default=None)

    parser.add_argument(
        "--gpu",
        action='store_true',
        help='switch to use GPU processing')

    # Actually get the argument values
    args = parser.parse_args()
    remote_host = args.remote_host
    input_file  = args.input_file
    window      = args.window
    layer_name  = args.layer
    gpu         = args.gpu 


    #print(args) # for debugging

    # Because XML-RPC calls can't handle named arguments, need to handle all
    # default values here...
    # TODO: check input file valid

    # TODO: check for valid window size
    if all(x == -1 for x in window):
        window = None


    # Set up the proxy server
    proxy = xmlrpclib.ServerProxy(remote_host, allow_none=True)

    # print(proxy.system.listMethods())
    feat = proxy.extract_features(input_file, layer_name, window, gpu)

    # To print to stdout
    for x in feat:
        print(x)

    base = os.path.basename(input_file)
    output_file = base + ".txt"
    np.savetxt(output_file, feat)

if __name__ == "__main__":
    main(sys.argv)