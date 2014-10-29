# Installing Caffe 

Very similar to installing for CPU only.

Using an `Ubuntu 14.04` image, on a g2.2xlarge machine.

**IMPORTANT: INSTALL CUDA FIRST**

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

Install CUDA
```
apt-get install linux-headers-`uname -r`

apt-get install build-essential

wget http://developer.download.nvidia.com/compute/cuda/6_5/rel/installers/cuda_6.5.14_linux_64.run

chmod +x cuda_6.5.14_linux_64.run

./cuda_6.5.14_linux_64.run --kernel-source-path=/usr/src/linux-headers-`uname -r`/
```

Press 'q' to skip to the end of the EULA, enter 'accept', 'n' to driver, 'y' on toolkit, `return` on location, 'y' on symlink, 'n' on samples.

Make it possible to find CUDA libraries
```
echo "/usr/local/cuda/lib64" > /etc/ld.so.conf.d/cuda.conf && \
  ldconfig 
```

Clone Caffe Repo
```
cd /opt && git clone https://github.com/BVLC/caffe.git
```

Install Glog
```
cd /opt && wget https://google-glog.googlecode.com/files/glog-0.3.3.tar.gz && \
  tar zxvf glog-0.3.3.tar.gz && \
  cd /opt/glog-0.3.3 && \
  ./configure && \
  make && \
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

Build Caffe Core
```
cd /opt/caffe && \
  cp Makefile.config.example Makefile.config && \
  echo "CXX := /usr/bin/g++-4.6" >> Makefile.config && \
  sed -i 's/CXX :=/CXX ?=/' Makefile && \
  make all
```

Install Python dependencies
```
cd /opt/caffe && easy_install numpy

cd /opt/caffe && pip install -r python/requirements.txt

easy_install pillow
```

Fix `libarray.h` error due to numpy not being found
```
NUMPY_EGG=`ls /usr/local/lib/python2.7/dist-packages | grep -i numpy`

ln -s /usr/local/lib/python2.7/dist-packages/$NUMPY_EGG/numpy/core/include/numpy /usr/include/python2.7/numpy
```

Build Python bindings for Caffe
```
cd /opt/caffe && make pycaffe
```

Make tests
```
cd /opt/caffe && make test
```

Run tests
```
cd /opt/caffe && make runtest
```

## Running MNIST LeNet Example

Depending on whether or not you're using Caffe w/ GPU, this could take a few minutes or a few hours...

```
cd /opt/caffe/data/mnist && ./get_mnist.sh

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
