% Setting up IPython Notebook server on EC2
% bab
% October 30, 2014

# Set EC2 Instance Security Group

You'll want to allow TCP traffic on the following ports: 22 (SSH), 443 (HTTPS), and 8888 for your instance. 

# Install Dependencies

If you're not using Anaconda, this can turn out to be surprisingly tedious. There are OS dependencies and Python dependencies, and pip may or may not actually decide to install the needed packages, if you do it in the wrong order.

Assuming the computer is set up according to `Install_CUDA_EC2.md` and `Install_GPU_AWS_EC2.md` or maybe just `Install_CPU_AWS_EC2`, you can proceed as follows

## Ubuntu 14.04 Dependencies

```bash
apt-get install -y libreadline-dev libncurses-dev libzmq-dev
```

You can install the most recent version of `libzmq` via:

```bash
apt-get install libtool pkg-config autoconf automake

wget download.zeromq.org/zeromq-4.1.0-rc1.tar.gz

tar zxvf zeromq-4.1.0-rc1.tar.gz

cd zeromq-4.1.0

./configure

make && make install
```

## IPython Notebook Dependencies

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

```
cd /home/ubuntu && mkdir -p certificates

cd /home/ubuntu/certificates 

openssl req -x509 -nodes -days 365 -newkey rsa:1024 -keyout nbserver_cert.pem -out nbserver_cert.pem
```

### Modify the IPython Notebook Config File

Basically, we want to: allow connections from any IP address, stop IPython from trying to open a browser, specify the port the server should listen on, point where the cert file is, and then specify a password hash we can use to authenticate. 

The differences between the default config and the config after these changes have been made look like this when using the `diff` tool:

```
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

# Notes

If, after setting up the server, and verifying that it is running, you get an error upon trying to connect, it is possible that you might just have to specify that you want to use HTTPS instead of HTTP. 

If you get a weird error regarding python egg security, you can try something like this to make it disappear (but perhaps not fix it)

```bash
chmod g-wx,o-wx ~/.python-eggs
```

