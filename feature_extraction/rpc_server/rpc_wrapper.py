#!python2.7
"""
A wrapper script which allows for command line feature extraction using Caffe,
with Caffe on a remote machine.

Assuming Caffe is running on a remote machine that you have access to, on your
local machine you would run this script via

    $ python rpc_wrapper.py <...args...>
"""

from __future__ import print_function

import argparse
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
        "--layer_name",
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
    layer_name  = args.layer_name
    gpu         = args.gpu 

    print(args) # REMOVE

    # Because XML-RPC calls can't handle named arguments, need to handle all
    # default values here...
    # TODO: check input file valid
    
    # TODO: check for valid window size
    if all(x == -1 for x in window):
        window = []


    # Set up the proxy server
    proxy = xmlrpclib.ServerProxy(remote_host)

    # print(proxy.system.listMethods())
    # feat = proxy.extract_features(input_file, layer_name, window)


if __name__ == "__main__":
    main(sys.argv)