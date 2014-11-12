# Installing Caffe for CPU-Only Processing on Ubuntu 14.04 

Having gone through this process a few times now, I think that it might now be possible to do all that is required in a slightly less arduous fashion, but at the expense of perhaps not getting the very latest versions of some packages...

Using an `Ubuntu 14.04` image

Running under root (`sudo su`)

## Update packages

```
apt-get update
apt-get upgrade
```


## Get Dependencies

```bash
apt-get install -y libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libboost-all-dev libhdf5-serial-dev protobuf-compiler gcc-4.6 g++-4.6 gcc-4.6-multilib g++-4.6-multilib gfortran libjpeg62 libfreeimage-dev libatlas-base-dev git python-dev python-pip bc wget curl unzip cmake liblmdb-dev pkgconf
```

The packages which are no longer manually compiled

```bash
apt-get install -y libgflags-dev libgoogle-glog-dev liblmdb-dev protobuf-compiler
```

## Add Swap Space 

If you're compiling things on a system with limited memory, it is sometimes helpful to
add some swap space. In particular, I've found that EC2 Micro Instances need swap space in order to compile `scipy` properly.

```
sudo /bin/dd if=/dev/zero of=/var/swap.1 bs=1M count=1024
sudo /sbin/mkswap /var/swap.1
sudo /sbin/swapon /var/swap.1
```

# IPython Notebook Installation Interlude

Assuming you want to get a notebook server running, you should perform the following actions before continuing.

## IPython Notebook Server Prerequisites

```
apt-get install -y libreadline-dev libncurses-dev libtool pkg-config autoconf automake
```

I think that getting the latest version of `libzmq` may be worthwhile, so use this

```bash
cd /home/ubuntu && mkdir -p tmp

cd /home/ubuntu/tmp && wget download.zeromq.org/zeromq-4.1.0-rc1.tar.gz

cd /home/ubuntu/tmp && tar zxvf zeromq-4.1.0-rc1.tar.gz

cd /home/ubuntu/tmp/zeromq-4.1.0 && ./configure

cd /home/ubuntu/tmp/zeromq-4.1.0 && make && make install
```

Then, install IPython's dependencies. It may be necessary to install Numpy first, so you might as well do it here:

```bash
easy_install numpy
```

```bash
pip install jinja2
pip install sphinx
pip install pyzmq
pip install pygments
pip install tornado
pip install nose
pip install readline
pip install ipython[all]
pip install ipython[notebook]
```

## Test

This is by no means definitive, but it can help identify glaring errors in how IPython is set up. Run the test suite:

```bash
iptest
```

## Create & Configure IPython Notebook Server Profile

### Create configuration files

```bash
ipython profile create nbserver 
```

### Create a password hash

Now, create a password hash so you can log in with moderately improved security.

```ipython
>>>from IPython.lib import passwd()
>>>passwd()
Enter password:
Verify password:

'sha1:4af286014356:47657f58db8615b025bab0703118d4106a314363'
```

The above corresponds to the password "nbserver"

### Create a self-signed SSL certificate

## Notes

If, after setting up the server, and verifying that it is running, you get an error upon trying to connect, it is possible that you might just have to specify that you want to use HTTPS instead of HTTP. 


# Installing Caffe

Clone the Caffe repo. 
```
cd /opt && git clone https://github.com/BVLC/caffe.git
cd /opt/caffe && git checkout 4288b2b5fc1fea600a336fc56fbaacaae5c94877
``` 

## Build Caffe Core

Build Caffe core, ensuring CPU_ONLY flag is set
```
cd /opt/caffe && cp Makefile.config.example Makefile.config
cd /opt/caffe && echo "CPU_ONLY := 1" >> Makefile.config 
cd /opt/caffe && echo "CXX := /usr/bin/g++-4.6" >> Makefile.config 
cd /opt/caffe && sed -i 's/CXX :=/CXX ?=/' Makefile
cd /opt/caffe && make all
```

## Install Python dependencies

It may be necessary to rerun the following, because of strange dependency handling in pip.

```
cd /opt/caffe && easy_install numpy

cd /opt/caffe && pip install -r python/requirements.txt

easy_install pillow
```

Create a symlink so that Caffe can find Numpy in the place it expects. *May have to change the path in the first part depending on the Numpy version*

```bash
ln -s /usr/local/lib/python2.7/dist-packages/numpy-1.9.1-py2.7-linux-x86_64.egg/numpy/core/include/numpy/ /usr/include/python2.7/numpy
```

## Build Python bindings

```bash
cd /opt/caffe && make pycaffe
```

## Make and Run Tests

```bash
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

# Optional Additional Steps

## Add Caffe to PYTHONPATH

In ~/.bashrc, add the line

```bash
export PYTHONPATH=$PYTHONPATH:/opt/caffe/python
```
