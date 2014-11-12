# Installing Caffe for CPU-Only Processing on Ubuntu 14.04 

Having gone through this process a few times now, I think that it might now be possible to do all that is required in a slightly less arduous fashion, but at the expense of perhaps not getting the very latest versions of some packages...

Using an `Ubuntu 14.04` image on Amazon EC2. It is likely that this will work on a fresh install of Ubuntu for whatever machine you're using, but I haven't tested that.

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

```bash
sudo /bin/dd if=/dev/zero of=/var/swap.1 bs=1M count=1024
sudo /sbin/mkswap /var/swap.1
sudo /sbin/swapon /var/swap.1
```

# IPython Notebook Installation Interlude

Assuming you want to get a notebook server running, you should perform the following actions before continuing. This leads to a different version of IPython being installed (one that includes the `notebook` subpackage), so it's best to do it before installing Caffe, although you can do it afterwards (or not at all!) without much additional hassle.

## IPython Notebook Server Prerequisites

There are a few things to get before we install everything else.

```bash
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

At this point, you should configure your IPython Notebook server for remote access. 

## Create Server Profile

We're going to use the IPython Profile `nbserver`, rather than modifying the default, so we need to create this profile and configure it. Once it's all said and done, we can run the notebook server via:

```bash
ipython notebook --profile=nbserver
```

### Create configuration files

```bash
ipython profile create nbserver 
```

### Create a password hash

```python
>>>from IPython.lib import passwd()
>>>passwd()
Enter password:
Verify password:

'sha1:4af286014356:47657f58db8615b025bab0703118d4106a314363'
```

The above corresponds to the password "nbserver"

### Create a self-signed SSL certificate

Create a directory to store the certificates, and then create your own. Note that if you try to navigate most browsers to a website with a self-signed certificate, they will complain; actually getting your own certificate is a process, but doable.

```bash
cd /home/ubuntu && mkdir -p certificates

cd /home/ubuntu/certificates 

openssl req -x509 -nodes -days 365 -newkey rsa:1024 -keyout nbserver_cert.pem -out nbserver_cert.pem
```

### Modify the IPython Notebook Config File

Basically, we want to: allow connections from any IP address, stop IPython from trying to open a browser, specify the port the server should listen on, point where the cert file is, and then specify a password hash we can use to authenticate. 

The differences between the default config and the config after these changes have been made look like this when using the `diff` tool:

```txt
18c18
< c.NotebookApp.ip = '*'
---
> # c.NotebookApp.ip = 'localhost'
42c42
< c.NotebookApp.open_browser = False
---
> # c.NotebookApp.open_browser = True
51c51
< c.NotebookApp.port = 8888
---
> # c.NotebookApp.port = 8888
84c84
< c.NotebookApp.certfile = u'/home/ubuntu/certificates/nbserver_cert.pem'
---
> # c.NotebookApp.certfile = u''
123c123
< c.NotebookApp.password = u'sha1:4af286014356:47657f58db8615b025bab0703118d4106a314363'
---
> # c.NotebookApp.password = u''
```

### Notes

If, after setting up the server, and verifying that it is running, you get an error upon trying to connect, it is possible that you might just have to specify that you want to use HTTPS instead of HTTP. 

If you get a weird error regarding python egg security, you can try something like this to make it disappear (but perhaps not fix it)

```bash
chmod g-wx,o-wx ~/.python-eggs
```

# Return to Installing Caffe

Clone the Caffe repo. 
```
cd /opt && git clone https://github.com/BVLC/caffe.git
cd /opt/caffe && git checkout 4288b2b5fc1fea600a336fc56fbaacaae5c94877
``` 

## Build Caffe Core

Build Caffe core, ensuring CPU_ONLY flag is set

```bash
cd /opt/caffe && cp Makefile.config.example Makefile.config
cd /opt/caffe && echo "CPU_ONLY := 1" >> Makefile.config 
cd /opt/caffe && echo "CXX := /usr/bin/g++-4.6" >> Makefile.config 
cd /opt/caffe && sed -i 's/CXX :=/CXX ?=/' Makefile
cd /opt/caffe && make all
```

## Install Python dependencies

It may be necessary to rerun the following, because of strange dependency handling in pip.

```bash
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
