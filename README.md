# README

This is the skeleton code for the 2017 Fall ECE408 / CS483 course project.
In this project, you will get experience with practical neural network artifacts, face the challenges of modifying existing real-world code, and demonstrate command of basic CUDA optimization techniques.
Specifically, you will get experience with

* Managing a simple machine-learning dataset.
* Using, profiling, and modifying MxNet a standard open-source neural-network framework.

You will demonstrate your CUDA expertise by

* Implementing an optimized neural network layer
* merged layer
* fp16
* anything else?

The project will be broken up into 3 milestones

## Milestone 1: Getting Started Due ()

### Getting Set Up

On your first clone, do 

    git submodule update --init --recursive

otherwise, you can pull an update version of the submodules with 

    git submodule update --recursive

### Install Prerequisites for Building `mxnet`.

The MxNet instructions are available [here](https://mxnet.incubator.apache.org/get_started/install.html). A short form of them follows.

    sudo apt install -y build-essential git libopenblas-dev liblapack-dev libopencv-dev

### Build mxnet 

Modify `make/config.mk`.

    USE_BLAS = openblas

If you want to use CUDA, you can do

    USE_CUDA = 1
    USE_CUDA_PATH = /usr/local/cuda

If you installed CuDNN, you can also do

    USE_CUDNN = 1

Then

    make

### Build Python Bindings

    sudo apt install -y python-dev python-setuptools python-numpy python-pip

Install the python binding

    cd python
    pip install --user -e .

You can always uninstall the python package with

    pip uninstall mxnet

This will uninstall anything installed with `--user` before anything else.

### Augment the fashion-mnist dataset

### Train and Test the base implementation

Adjust your performance expectations based on whether you're using CUDA or CUDNN.

| Context  | Performance  |
|---|---|
| i7-5820  | 450 images/sec  |
| GTX 1070  | 8k images/sec  |
| GTX 1070 w/cudnn  | 50k images/sec  | 

You should achieve an accuracy of XXX after XXX iterations.

### Generate a NVPROF Profile

Once you've gotten the appropriate accuracy results, generate a profile.

    nvprof fashion-mnist.py

## Milestone 2: A New Convolution Layer in MxNet

    https://mxnet.incubator.apache.org/get_started/install.html

This README would normally document whatever steps are necessary to get your application up and running.





## Final Submission: An optimized layer

## Extras

### Setting up Visual Studio Code

    sudo apt install python-pip
    pip install --user quilt numpy pylint pep8 autopep8