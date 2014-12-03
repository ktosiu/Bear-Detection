#!python3
"""
Sliding windows as region proposals

Superpixels for taking windows of different sizes.
"""

import numpy as np 
from numpy.lib.stride_tricks import as_strided


def sliding_window(a, ws, ss=None):
    """
    Compute n-dimensional sliding windows.

    Parameters
    ----------
    a  : numpy.ndarray
    ws : int or iterable of int
        window size
    ss : int or iterable of int
        stride size 
    """
    if ss is None:
        ss = ws 
    
    # Would do validtion here
    
    # Convert `ss`, `ws`, and `a.shape` to numpy arrays
    ws = np.array(ws)
    ss = np.array(ss)
    shape = np.array(a.shape)

    new_shape = ((shape - ws) // ss) + 1