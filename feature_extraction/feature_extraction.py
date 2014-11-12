#!python
"""
Example of how to perform feature extraction
"""

# As the output of forward is a dictionary whose values are arrays containing
# results of various layers, which are update whenever predictions are made.
out = net.forward()

