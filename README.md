# README

## Introduction

This is the skeleton code for the 2017 Fall ECE408 / CS483 course project.
In this project, you will get experience with practical neural network artifacts, face the challenges of modifying existing real-world code, and demonstrate command of basic CUDA optimization techniques.
Specifically, you will get experience with

* Using, profiling, and modifying MxNet, a standard open-source neural-network framework.

You will demonstrate command of CUDA and optimization approaches by

* Implementing an optimized neural-network convolution layer forward pass

The project will be broken up into 3 milestones and a final submission. Read the description of the final report before starting, so you can collect the necessary info along the way.

## Deliverables Overview

You may wish to stay ahead of these deadlines (particularly, allow more than two weeks between milestone 3 and the final submission).

1. [Milestone 1: Getting Started: Due 11/10/2017](#markdown-header-milestone-1)
    1. [Run the MXNet baseline forward CPU pass.]()
    2. [Run the MXNet baseline forward GPU pass.]()
    3. [Generate a profile of the GPU forward pass using `nvprof`.]()
2. [Milestone 2: A New CPU Layer in MXNet: Due 11/17/2017](#markdown-header-milestone-2)
    1. [Implement a CPU convolution pass in MXNet]()
2. [Milestone 3: A New GPU Layer in MXNet: Due 12/1/2017](#markdown-header-milestone-3)
    1. [Implement a GPU convolution in MXNet]()
3. [Final Submission: Optimized GPU Forward Implementation: Due 12/15/2017](#markdown-header-final-submission)
    1. [Implement an optimized GPU convolution in MXNet]()
    2. [Final Report](#markdown-header-final-report)

## Remote Development Environment

The easiest way to develop the project is to use rai through the following prebuilt binaries. You can also use the Linux machines on [EWS](http://it.engineering.illinois.edu/ews) for RAI.

**NOTE:** Even if you use your own local development environment, your final code must run within the RAI system. 

See the [Client Documentation Page](https://github.com/rai-project/rai) for information on how to download, setup, and use the client on your own laptop or desktop.

## Milestone 1
**Getting Started: Due Friday November 10th, 2017**

### Getting Set Up and Getting Bugfixes

Clone this repository to get the project directory.

    git clone https://cwpearson@bitbucket.org/hwuligans/2017fa_ece408_project.git

We suggest using rai to develop your project. **You will use rai to submit your project**.

### 1.1: Run the Baseline Forward Pass

The neural network architecture used for this project is shown below.

| Layer |       Desc      |
|-------|-----------------|
| 0     | input           |
| 1     | convolution     |
| 2     | tanh            |
| 3     | pooling         |
| 4     | convolution     |
| 5     | tanh            |
| 6     | pooling         |
| 7     | fully connected |
| 8     | tanh            |
| 9     | fully connected |
| 10    | softmax         |

Use RAI to run a batch forward pass on some test data.

    rai

This will upload your project directory to rai (running on an IBM 8335-GTB "Minsky") and move it to `/src`, where the execution specified in `rai_build.yml` will occur. For this CPU run, leave `rai_build.yml` untouched, but look at its contents.

The `image:` key specifies the environment that the rest of the execution will occur in.
This environment includes a prebuilt MXNet (so rai will only do a partial compile with your code) as well as the model definition and the training data.

The `resources:` key specifies what computation resources will be available to the execution.

The `commands:` key specifies the recipe that rai will execute. `./build.sh` copies the files in `ece408_src` to `src/operator/custom/` in the MXNet source tree, and then compiles MXNet and installs the MXNet python bindings into the environment.
You do not need to modify `build.sh` to successfully complete the project, but look at it if you are curious.
`python /src/m1.1_forward_mxnet_conv.py` runs the `m1.1_forward_mxnet_conv.py` python program.

You should see the following output:

    output
    output
    output

The accuracy should be exactly XXX. 
There is no specific deliverable for this portion.

### 1.2: Run the baseline GPU implementation

To switch to a GPU run, you will need to modify rai_build.yml.

| original line | replacement line | 
| -- | -- | 
| `image: cwpearson/2017fa_ece408_mxnet_docker:ppc64le-cpu-latest` | `image: cwpearson/2017fa_ece408_mxnet_docker:ppc64le-gpu-latest` |
| `count: 0` | `count: 1` |
| `python /src/m1.1_forward_mxnet_conv.py` | `python /src/m1.2_forward_mxnet_conv.py` |

This uses a rai environment with MXNet built for CUDA, tells rai to use a GPU, and runs `m1.2_forward_mxnet_conv.py` instead of `m1.1_forward_mxnet_conv.py`.

Compare `m1.2_forward_mxnet_conv.py` and `m1.1_forward_mxnet_conv.py`. You'll see that it is the same, except for `mx.gpu()` has been substituted for `mx.cpu()`. This is how we tell MXNet that we wish to use a GPU instead of a CPU.

Again, submit to rai

    rai

You should see the same accuracy as the CPU version. 
There is no specific deliverable for this portion.

### 1.3 Generate a NVPROF Profile

Once you've gotten the appropriate accuracy results, generate a profile using nvprof. You will be able to use nvprof to evaluate how effective your optimizations are.
As described above, make sure `rai_build.yml` is configured for a GPU run.
Then, modify `rai_build.yml` to generate a profile instead of just execuing the code.

    nvprof python m1.2_forward_mxnet_conv.py

You should see something that looks like the following:

    output 
    output
    output

You can see how much time MXNet is spending on a variety of the operators. Look for `XXX` and report the cumulative time that MXNet spends on that operation.

## Milestone 2
**A New CPU Convolution Layer in MxNet: Due Friday November 17th, 2017**

See the [description](#markdown-header-skeleton-code-description) of the skeleton code for background information.

### 2.1 Add a simple CPU forward implementation

Modify `ece408_src/new-forward.h` to implement the forward convolution described in [Chapter 16 of the textbook](https://wiki.illinois.edu/wiki/display/ECE408Fall2017/Textbook+Chapters).
The performance of the CPU convolution is not part of the project evaluation.

## Milestone 3
**A New GPU Convolution Layer in MxNet: Due Friday December 1st, 2017**

### 3.1 Add a simple GPU forward implementation

Modify `ece408_src/new-forward.cuh` to implement a forward GPU convolution.

### 3.2 Create a profile with `nvprof`.

Provide a profile showing that the forward pass is running on the GPU.
You should see output that looks something like this:

    output
    output
    output

## Final Submission
**An Optimized Layer and Final Report: Due Friday December 15th, 2017**

### Optimized Layer

Optimize your GPU convolution.

### Final Report

You should provide a brief PDF final report, with the following content.

1. **Milestone 1**
    1. built-in CPU performance results
        1. execution time
    2. built-in GPU performance results
        1. execution time
        2. `nvprof` profile
2. **Milestone 2**
    1. baseline solution CPU performance results
        1. execution time
3. **Optimization Approach**
    * how you identified the optimization opportunity
    * why you thought the approach would be fruitful
    * the effect of the optimization. was it fruitful, and why or why not. Use nvprof as needed
    * Any external references used during identification or development of the optimization
4. **References** (as needed)

Do not make your report longer than it needs to 

## Skeleton Code Description

`new-forward.h` and `new-forward.cuh` contain skeleton implementations for CPU and GPU convolutions. You can complete the project by modifying only these two files. These functions are called from `Forward()` in `new-inl.h`.

The code in `new-inl.h`, `new.cc`, and `new.cu` describes the convolution layer to MXNet. You will not need to modify these files, though you can if you want to.

| File | Function | Description |
| -- | -- | -- |
| `new-forward.h` | `forward()` | Your CPU implementation goes here. |
| `new-forward.cuh` | `forward()` | Your GPU host code goes here. |
| `new-forward.cuh` | `forward_kernel()` | Your GPU kernel implementation goes here. |
| -- | -- | -- |
| `new-inl.h` | `InferShape()` | Computes shape of output tensor from input and kernel shape |
| `new-inl.h` | `InferType()` | Computes type of the output tensor based on the inputs. |
| `new-inl.h` | `Forward()` | Defines the operations of the forward pass. Calls our implementation. |
| `new-inl.h` | `Backward()` | Defines the operations of the backward (training) pass. Not used in this project. |
| `new-inl.h` | `struct NewParam` | Defines the arguments passed to the operator in python. |
| `new.cc` | `CreateOperatorEx()` | Called by MXNet to create the appropriate operator for a CPU or GPU execution. |
| `new.cc` | `CreateOp<cpu>()` | Creates the CPU operator. |
| `new.cu` | `CreateOp<gpu>()` | Creates the GPU operator when CUDA is enabled. |

## Extras

### Multiple Datasets

We will be checking final submissions on a dataset you are not provided.
To check your implementation, you can use the two provided datasets in `/models`.

* `/models/ece408-low` for a low-accuracy model (accuracy = 0.6964 for `t10k`)
* `/models/ece408-high` for a high-accuracy model ( accuracy = 0.8458 for `t10k`)

The result should be the same for both the CPU and GPU convolutions.

### Checking for Errors

Within MXNet, you can use `MSHADOW_CUDA_CALL(...);` as is done in `new-forward.cuh`.
Or, you can define a macro/function similar to `wbCheck` used in WebGPU.

### Comparing GPU implementation to CPU implementation

It may be hard to directly debug by inspecting values during the forward pass since the weights are already trained and the input data is from a real dataset.
You can always extract your implementations into a separate set of files, generate your own test data, and modify `rai_build.yml` to build execute your separate test code instead of the MXNet code while developing.

**None of the following is needed to complete the course project.**

If you'd like to develop using a local copy of mxnet, you may do so. Keep in mind your project will be evaluated through rai. Your submission must work through rai.

The MxNet instructions are available [here](https://mxnet.incubator.apache.org/get_started/install.html). A short form of them follows for Ubuntu.

    # install some prereqs
    sudo apt install -y build-essential git libopenblas-dev liblapack-dev libopencv-dev python-pip python-dev python-setuptools python-numpy
    # download mxnet release 0.11.0
    git clone git@github.com:apache/incubator-mxnet.git --recursive --branch 0.11.0
    # build mxnet
    nice -n20 make -C incubator-mxnet -j$(nrpoc) USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1 USE_BLAS=openblas
    # install python bindings
    pip install --user -e incubator-mxnet/python

You can always uninstall the python package with

    pip uninstall mxnet

Download the fashion-mnist dataset

    mkdir fashion-mnist \
    && wget -P fashion-mnist \
        http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz \
        http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz \
        http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz \
        http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz

Modify the python forward convolution scripts to point to where you downloaded fashion-mnist

    ... load_mnist(path="fashion-mnist", ...)

Download the trained models (for the existing mxnet implementation and your implementation)

    mkdir -p models \
    && wget -P models \
        https://github.com/cwpearson/2017fa_ece408_mxnet_docker/raw/master/models/baseline-0001.params \
        https://github.com/cwpearson/2017fa_ece408_mxnet_docker/raw/master/models/baseline-symbol.json \
        https://github.com/cwpearson/2017fa_ece408_mxnet_docker/raw/master/models/ece408-high-0001.params \
        https://github.com/cwpearson/2017fa_ece408_mxnet_docker/raw/master/models/ece408-high-symbol.json \
        https://github.com/cwpearson/2017fa_ece408_mxnet_docker/raw/master/models/ece408-low-0001.params \
        https://github.com/cwpearson/2017fa_ece408_mxnet_docker/raw/master/models/ece408-low-symbol.json

Modify the python forward convolution scripts to point to where you downloaded fashion-mnist

    lenet_model = mx.mod.Module.load( prefix='models/baseline' ... 

Modify `build.sh` to point at your mxnet code.

    ...
    MXNET_SRC_ROOT=<your incubator-mxnet path here>
    ...


## License

NCSA/UIUC © [Carl Pearson](https://cwpearson.github.io)