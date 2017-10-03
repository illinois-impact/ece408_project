# README

This is the skeleton code for the 2017 Fall ECE408 / CS483 course project.
In this project, you will get experience with practical neural network artifacts, face the challenges of modifying existing real-world code, and demonstrate command of basic CUDA optimization techniques.
Specifically, you will get experience with

* Using, profiling, and modifying MxNet, a standard open-source neural-network framework.

You will demonstrate command of CUDA and optimization approaches by

* Implementing an optimized neural-network convolution layer forward pass

The project will be broken up into 3 milestones. Read the description of the final report before starting, so you can collect the necessary info along the way.

## Deliverables Overview

1. [Milestone 1: Getting Started: Due ()](#markdown-header-milestone-1)
    1. [Train the baseline network on the CPU.]()
    2. [Train the baseline network on the GPU.]()
    3. [Generate a profile of the GPU training using `nvprof`.]()
2. [Milestone 2: A New Layer in MXNet: Due ()](#markdown-header-milestone-2)
    1. []()
3. [Final Submission 3: Optimized GPU Forward Implementation](#markdown-header-milestone-3)
    1. []()
    2. [Final Report](#markdown-header-final-report)

## Milestone 1
**Getting Started: Due ()**

### Getting Set Up and Getting Bugfixes

Clone this repository.

    git clone https://cwpearson@bitbucket.org/hwuligans/2017fa_ece408_project.git

This will put you on the `master` branch. There may be unstable "improvements" in the `develop` branch of this repository.

You will be using rai to develop and submit your project.

### (optional) Augment the fashion-mnist dataset

### 1.1 Train the baseline CPU implementation

A simple convolutional neural network is implemented in `fashion-mnist.py`.
Read the comments in that file to understand the structure of the network.
Check that `fashion-mnist.py` is using the CPU, and using the built-in MXNet convolution, and only training for 1 epoch:

    conv1 = mx.sym.Convolution(...
    # ...
    conv2 = mx.sym.Convolution(...
    # ...
    lenet_model = ... context=mx.cpu())
    # ...
    ... num_epoch=1)

Train  the `fashion-mnist` network for one epoch by submitting the job to RAI

    rai

This will execute the actions in `rai_build.yml`. The `image:` key in `rai_build.yml` specifies the environment that the rest of the execution will occur in. That environment has a pre-build MXnet, so you will not need to wait on a full rebuild every time you submit to rai.

The contents of this directory will be sent to the RAI backend (running on an IBM 8335-GTB "Minsky"). `build.sh` will be executed, and then `python fashion-mnist.py`.

`build.sh` copies the files in `ece408_src` to `src/operator/custom/` in the MXNet source tree in the rai environment, and rebuilds MXNet to include your new code. It then installs the Python bindings into the environment.

You should achieve an accuracy of XXX after the single epoch finishes.

### 1.2 Train the baseline GPU implementation

The baseline GPU implementation is much faster. Modify `fashion-mnist.py` to train for a few more epochs, and execute on the GPU:

    lenet_model = ... context=mx.gpu())
    # ...
    ... num_epoch=10)

Again, submit to rai

    rai

You should see much greater performance, and again an accuracy of XXX after the training is done.

### 1.3 Generate a NVPROF Profile

Once you've gotten the appropriate accuracy results, generate a profile. Modify `rai_build.yml` to generate a profile instead of just execuing the code.

    nvprof python fashion-mnist.py

## Milestone 2
**A New Convolution Layer in MxNet: Due ()**

### 2.1 Add a simple CPU forward implementation

Modify `src/operator/custom/ece408.cc` to implement the forward CPU operator. 

## Final Submission: An optimized layer

### Optimized Layer

### Final Report
**Due ()**

You should provide a brief final report, with the following content.

1. Milestone 1
    1. built-in CPU performance results
        1. execution time
    2. built-in GPU performance results
        1. execution time
        2. `nvprof` profile
2. Milestone 2
    1. baseline solution CPU performance results
        1. execution time
3. **Optimization Approach**
    * how you identified the optimization opportunity
    * why you thought the approach would be fruitful
    * the effect of the optimization. was it fruitful, and why or why not. Use nvprof as needed
    * Any external references used during identification or development of the optimization
4. References (if needed)

Do not make your report longer than it needs to 

## Extras

**None of the following is needed to complete the course project.**

If you'd like to develop using a local copy of mxnet, you may do so. Keep in mind your project will be evaluated through rai. Your submission must work through rai.

The MxNet instructions are available [here](https://mxnet.incubator.apache.org/get_started/install.html). A short form of them follows for Ubuntu.

    # install some prereqs
    sudo apt install -y build-essential git libopenblas-dev liblapack-dev libopencv-dev python-pip python-dev python-setuptools python-numpy
    # download mxnet release 0.11.0
    git clone git@github.com:apache/incubator-mxnet.git --recursive --branch 0.11.0
    # build mxnet
    nice -n20 make -C incubator-mxnet -j$(nrpoc) USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_OPENCV=1 USE_CUDNN=1 USE_BLAS=openblas
    # install python bindings
    pip install --user -e incubator-mxnet/python

You can always uninstall the python package with

    pip uninstall mxnet

Download the fashion-mnist dataset


    mkdir fashion_mnist
    wget -P fashion_mnist \
        http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz \
        http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz \
        http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz \
        http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz

Modify `test_fashion_mnist.py` to point to where you downloaded fashion-mnist

    ... load_mnist(path="fashion-mnist", ...)

Modify `build.sh` to point at your mxnet code.

    ...
    MXNET_SRC_ROOT=<your incubator-mxnet path here>
    ...

If you want to experiment with training or defining a different network architecture (neither is part of the project), you can use `train_fashion_mnist.py`. Some example performance numbers using the built-in MXNet operators.

| Context  | Performance  |
|---|---|
| (CPU) Core i7-5820k    | 450 images/sec  |
| (GPU) GTX 1070         | 8k images/sec   |
| (GPU) GTX 1060 w/cudnn | 14k images/sec  |
| (GPU) GTX 1070 w/cudnn | 70k images/sec  | 