Here, in no particular order, are a series of problems I have encountered, and some of the things that have helped fix them.

# cannot import caffe

Make sure that Caffe has been added to `PYTHONPATH`. In `.bashrc`, add a line at the end:
```
export PYTHONPATH=$PYTHONPATH:/opt/caffe/python
```

# error: Failed to initialize libdc1394

There's no authoritative answer; it's possible that there's some sort of driver missing because we're running in a virtual machine. [StackOverflow suggested](https://stackoverflow.com/questions/12689304/ctypes-error-libdc1394-error-failed-to-initialize-libdc1394):

```bash
sudo ln /dev/null /dev/raw1394
```

# running out of memory during compile

## Scipy failing to compile

It could be because the system is running out of memory. To add some swap space, you can use:

```
sudo /bin/dd if=/dev/zero of=/var/swap.1 bs=1M count=1024
sudo /sbin/mkswap /var/swap.1
sudo /sbin/swapon /var/swap.1
```

# Fixing failures to load kernel module `nvidia.ko`

Via: https://tleyden.github.io/blog/2014/10/25/cuda-6-dot-5-on-aws-gpu-instance-running-ubuntu-14-dot-04/

```
apt-get install linux-image-extra-virtual
```