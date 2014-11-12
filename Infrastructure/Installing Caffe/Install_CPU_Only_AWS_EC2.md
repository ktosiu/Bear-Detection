% Installing CPU-Only Version of Caffe on EC2 
% bab
% October 30, 2014

# Installing Caffe

This takes a significant amount of time. The purpose of a number of these libraries and so on that are unknown to me. It compiles more or less everything needed, which can take awhile. It might be better to use a Python package/distribution manager like Anaconda, but I haven't gotten that running yet. 

## CPU Only (No GPU)

Following the build process here: https://registry.hub.docker.com/u/tleyden5iwx/caffe/dockerfile/

Using an `Ubuntu 14.04` image

Running under root (`sudo su`)

Update packages

```
apt-get update
apt-get upgrade
```

Get dependencies
```
apt-get install -y libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libboost-all-dev libhdf5-serial-dev protobuf-compiler gcc-4.6 g++-4.6 gcc-4.6-multilib g++-4.6-multilib gfortran libjpeg62 libfreeimage-dev libatlas-base-dev git python-dev python-pip bc wget curl unzip cmake liblmdb-dev pkgconf
```

Use GCC 4.6
```
update-alternatives --install /usr/bin/cc cc /usr/bin/gcc-4.6 30
update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++-4.6 30 
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.6 30 
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.6 30
```

Clone the Caffe repo. Note that this build uses an older version...
```
cd /opt && git clone https://github.com/BVLC/caffe.git
cd /opt/caffe && git checkout 4288b2b5fc1fea600a336fc56fbaacaae5c94877
```

Install Glog
```
cd /opt && wget https://google-glog.googlecode.com/files/glog-0.3.3.tar.gz
cd /opt && tar zxvf glog-0.3.3.tar.gz 
cd /opt/glog-0.3.3 
./configure 
make 
make install
```

To fix errors loading libglog, run `ldconfig`. May need to actively set the `LD_LIBRARY_PATH` environment variable, or run `ldconfig` in `~/.bashrc`

```
ldconfig
```

Install Gflags
```
cd /opt && wget https://github.com/schuhschuh/gflags/archive/master.zip
cd /opt && unzip master.zip
cd /opt/gflags-master && mkdir build 
cd /opt/gflags-master/build
export CXXFLAGS="-fPIC" 
cmake .. 
make VERBOSE=1
make 
make install
```

Build Caffe core, ensuring CPU_ONLY flag is set
```
cd /opt/caffe && cp Makefile.config.example Makefile.config
cd /opt/caffe && echo "CPU_ONLY := 1" >> Makefile.config 
cd /opt/caffe && echo "CXX := /usr/bin/g++-4.6" >> Makefile.config 
cd /opt/caffe && sed -i 's/CXX :=/CXX ?=/' Makefile
cd /opt/caffe && make all
```

Install Python dependencies
```
cd /opt/caffe && easy_install numpy

cd /opt/caffe && pip install -r python/requirements.txt

easy_install pillow
```

Create a symlink so that Caffe can find Numpy in the place it expects. *May have to change the path in the first part depending on the Numpy version*
```
ln -s /usr/local/lib/python2.7/dist-packages/numpy-1.8.2-py2.7-linux-x86_64.egg/numpy/core/include/numpy /usr/include/python2.7/numpy
```

Build the Python bindings
```
cd /opt/caffe && make pycaffe
```

Make and run tests
```
cd /opt/caffe && make test && make runtest
```

# Running examples

## Running MNIST LeNet Example

Depending on whether or not you're using Caffe w/ GPU, this could take a few minutes or a few hours...

```
cd /opt/caffe/data/mnist && ./get_mnist.sh

cd /opt/caffe/examples/mnist 

sed -i 's/solver_mode: GPU/solver_mode: CPU/' lenet_solver.prototxt

cd /opt/caffe && ./examples/mnist/create_mnist.sh

cd /opt/caffe && ./examples/mnist/train_lenet.sh
```

## Testing that Python works w/ Caffe

Get necessary files
```
cd /opt/caffe

./data/ilsvrc12/get_ilsvrc_aux.sh 

./scripts/download_model_binary.py models/bvlc_reference_caffenet
```

Test classification *does not work with `--print_results`*
```
python python/classify.py --print_results examples/images/cat.jpg foo
```

# Fixing Random Problems

## Fixing libdc1394 error

I found success by just linking `/dev/raw1394` to `/dev/null`

```
sudo ln /dev/null /dev/raw1394
```

# Guides

## Installing Caffe

Some of these may not have been used, but they can give a general idea of what needs to be done. 

https://tleyden.github.io/blog/2014/10/25/cuda-6-dot-5-on-aws-gpu-instance-running-ubuntu-14-dot-04/

https://registry.hub.docker.com/u/tleyden5iwx/caffe-gpu/

https://registry.hub.docker.com/u/tleyden5iwx/caffe/dockerfile/

https://github.com/BVLC/caffe/wiki/Ubuntu-14.04-VirtualBox-VM

https://github.com/BVLC/caffe/wiki/Ubuntu-14.04-ec2-instance

https://github.com/BVLC/caffe/issues/1092

http://petewarden.com/2014/07/25/setting-up-caffe-on-ubuntu-14-04/

https://gist.github.com/mitmul/217d26bd028b0e12c771