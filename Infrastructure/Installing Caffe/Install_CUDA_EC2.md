% Installing CUDA on EC2 
% bab
% October 30, 2014

# Installing Cuda

On a g2.2xlarge instance running Ubuntu Server 14.04 LTS (HVM) SSD Volume Type

Become root:
```
sudo su
```

Update and upgrade
```
apt-get -y update &&  apt-get -y upgrade
```
```
apt-get install -y build-essential
```

Install what is presumably virtual machine driver support
```
sudo apt-get install -y linux-image-extra-virtual
```

Reboot.

Disable nouveau, since it conflicts with the nvidia kernel module.

Open a file
```
nano /etc/modprobe.d/blacklist-nouveau.conf
```

Add these lines to it
```
blacklist nouveau
blacklist lbm-nouveau
options nouveau modeset=0
alias nouveau off
alias lbm-nouveau off
```

Save the file.

Disable the Kernel Nouveau
```
echo options nouveau modeset=0 | sudo tee -a /etc/modprobe.d/nouveau-kms.conf
```

Reboot again, after `update-initramfs -u`
```
update-initramfs -u

reboot
```

Get kernel source
```
apt-get install linux-source

apt-get install linux-headers-`uname -r`
```

Get CUDA installer
```
wget http://developer.download.nvidia.com/compute/cuda/6_5/rel/installers/cuda_6.5.14_linux_64.run
```

Extract CUDA installer
```
chmod +x cuda_6.5.14_linux_64.run

mkdir nvidia_installers

./cuda_6.5.14_linux_64.run -extract=`pwd`/nvidia_installers
```

 Run Nvidia driver installer
```
 cd nvidia_installers

./NVIDIA-Linux-x86_64-340.29.run
```

Load nvidia kernel module
```
modprobe nvidia
```

Run CUDA and samples installer
```
./cuda-linux64-rel-6.5.14-18749181.run

./cuda-samples-linux-6.5.14-18745345.run
```

Verify that CUDA is correctly installed
```
cd /usr/local/cuda/samples/1_Utilities/deviceQuery

make

./deviceQuery
```

Should see something like:
```
... lots of text ...

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 6.5, CUDA Runtime Version = 6.5, NumDevs = 1, Device0 = GRID K520
Result = PASS
```
