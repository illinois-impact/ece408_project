# README

## Introduction

This is the skeleton code for the 2017 Fall ECE408 / CS483 course project.
In this project, you will get experience with practical neural network artifacts, face the challenges of modifying existing real-world code, and demonstrate command of basic CUDA optimization techniques.
Specifically, you will get experience with

* Using, profiling, and modifying MxNet, a standard open-source neural-network framework.

You will demonstrate command of CUDA and optimization approaches by

* Implementing an optimized neural-network convolution layer forward pass

The project will be broken up into 3 milestones and a final submission. Read the description of the final report before starting, so you can collect the necessary info along the way.

You will be working in teams of 3.

You are expected to adhere to University of Illinois Academic integrity standards.
Do not attempt to subvert and of the performance-measurement aspects of the final project.
If you are unsure about whether something does not meet those guidelines, ask a member of the teaching staff.

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

## Milestone 1
**Getting Started: Due Friday November 10th, 2017**

Nothing must be turned in for this milestone, but this contributes to the final report.

### Getting Set Up and Getting Bugfixes

Clone this repository to get the project directory.

    git clone https://cwpearson@bitbucket.org/hwuligans/2017fa_ece408_project.git

We suggest using rai to develop your project. **You will use rai to submit your project**.

Download the rai binary for your platform

| Operating System | Architecture | Stable Version Link                                                             |
| ---------------- | ------------ | ------------------------------------------------------------------------------- |
| Linux            | amd64        | [URL](http://files.rai-project.com/dist/rai/stable/latest/linux-amd64.tar.gz)   |
| Linux            | ppc64le      | [URL](http://files.rai-project.com/dist/rai/stable/latest/linux-ppc64le.tar.gz) |
| OSX/Darwin       | amd64        | [URL](http://files.rai-project.com/dist/rai/stable/latest/darwin-amd64.tar.gz)  |
| Windows          | amd64        | [URL](http://files.rai-project.com/dist/rai/stable/latest/windows-amd64.tar.gz) |

You should have received a `.rai_profile` file by email.
Put that file in `~/.rai_profile` (Linux/macOS) or `%HOME%/.rai_profile` (Windows).
As soon as you and your two teammates agree on a team name, fill in the corresponding entry in your `.rai_profile`.
**Be sure you all use the same team name**.

Some more info is available on the [Client Documentation Page](https://github.com/rai-project/rai).

### 1.1: Run the Baseline Forward Pass

**Goal: Run CPU code in rai**

The neural network architecture used for this project is shown below.

| Layer |       Desc      |
|-------|-----------------|
| 0     | input           |
| 1     | convolution     |
| 2     | tanh            |
| 3     | pooling         |
| 4     | fully connected |
| 5     | tanh            |
| 6     | fully connected |
| 7     | softmax         |

Use RAI to run a batch forward pass on some test data.

    rai -p <project-folder>

This will upload your project directory to rai (running on an IBM 8335-GTB "Minsky") and move it to `/src`, where the execution specified in `rai_build.yml` will occur. For this CPU run, leave `rai_build.yml` untouched, but look at its contents.

The `image:` key specifies the environment that the rest of the execution will occur in.
This environment includes a prebuilt MXNet (so rai will only do a partial compile with your code) as well as the model definition and the training data.

The `resources:` key specifies what computation resources will be available to the execution.

The `commands:` key specifies the recipe that rai will execute. `./build.sh` copies the files in `ece408_src` to `src/operator/custom/` in the MXNet source tree, and then compiles MXNet and installs the MXNet python bindings into the environment.
You do not need to modify `build.sh` to successfully complete the project, but look at it if you are curious.
`python /src/m1.1_forward_mxnet_conv.py` runs the `m1.1_forward_mxnet_conv.py` python program.

You should see the following output:

    Loading fashion-mnist data... done
    Loading model... done
    EvalMetric: {'accuracy': 0.8673}

There is no specific deliverable for this portion.

### 1.2: Run the baseline GPU implementation

**Goal: Run GPU code in rai**

To switch to a GPU run, you will need to modify rai_build.yml.

| original line | replacement line | 
| -- | -- | 
| `image: cwpearson/2017fa_ece408_mxnet_docker:amd64-cpu-latest` | `image: cwpearson/2017fa_ece408_mxnet_docker:amd64-gpu-latest` |
| `count: 0` | `count: 1` |
| `python /src/m1.1.py` | `python /src/m1.2.py` |

This uses a rai environment with MXNet built for CUDA, tells rai to use a GPU, and runs `m1.2.py` instead of `m1.1.py`.

Compare `m1.2.py` and `m1.1.py`. You'll see that it is the same, except for `mx.gpu()` has been substituted for `mx.cpu()`. This is how we tell MXNet that we wish to use a GPU instead of a CPU.

Again, submit to rai

    rai -p <project-folder>

You should see the same accuracy as the CPU version. 
There is no specific deliverable for this portion.

### 1.3 Generate a NVPROF Profile

**Goal: Be able to use nvprof for performance evaluation**

Once you've gotten the appropriate accuracy results, generate a profile using nvprof. You will be able to use nvprof to evaluate how effective your optimizations are.
As described above, make sure `rai_build.yml` is configured for a GPU run.
Then, modify `rai_build.yml` to generate a profile instead of just execuing the code.

    nvprof python m1.2.py

You should see something that looks like the following:

    ✱ Running nvprof python /src/m1.2_forward_mxnet_conv.py
    Loading fashion-mnist data... done
    ==308== NVPROF is profiling process 308, command: python     /src/m1.2_forward_mxnet_conv.py
    Loading model... done
    EvalMetric: {'accuracy': 0.8673}
    ==308== Profiling application: python /src/m1.2_forward_mxnet_conv.py
    ==308== Profiling result:
    Time(%)      Time     Calls       Avg       Min       Max  Name
     30.77%  8.7488ms         1  8.7488ms  8.7488ms  8.7488ms  sgemm_128x128x8_NT_vec
     24.47%  6.9571ms        13  535.16us     480ns  5.8152ms  [CUDA memcpy HtoD]
     16.39%  4.6598ms         2  2.3299ms  92.225us  4.5676ms  void
     ... < snip > ...
    
    ==308== API calls:
    Time(%)      Time     Calls       Avg       Min       Max  Name
     52.04%  4.08135s        10  408.14ms  1.1290us  1.02416s  cudaFree
     37.08%  2.90862s        16  181.79ms  26.465us  1.45398s  cudaStreamCreateWithFlags
      9.95%  780.12ms        24  32.505ms  316.37us  768.95ms  cudaMemGetInfo
    ... < snip > ...



You can see how much time MXNet is spending on a variety of the operators.

## Milestone 2
**A New CPU Convolution Layer in MxNet: Due Friday November 17th, 2017**

A draft of the `report.pdf` with content up through Milestone 2 must be submitted **through rai** for this milestone.

See the [description](#markdown-header-skeleton-code-description) of the skeleton code for background information, including the data storage layout of the tensors.

### 2.1 Add a simple CPU forward implementation

**Goal: successfully edit code and run in rai**

Modify `ece408_src/new-forward.h` to implement the forward convolution described in [Chapter 16 of the textbook](https://wiki.illinois.edu/wiki/display/ECE408Fall2017/Textbook+Chapters).
The performance of the CPU convolution is not part of the project evaluation.

Because this operator is different than the built-in mxnet operator, you will need to load a different model.
`m2.1.py` handles this for you.
Modify rai_build.yml to invoke

    python m2.1py

When your implementation is correct, you should see output like this:

    ✱ Running python /src/m2.1.py
    Loading fashion-mnist data... done
    Loading model... done
    Time: 12.819000
    Correctness: 0.8562 Batch Size: 10000 Model: ece408-high

`m2.1.py` takes two position arguments. The first is the model name, the second is the dataset size. 
If the correctness for each possible model is as below, you can be reasonably confident your implementation is right. 
The correctness does depend on the data size. Check your correctness on the full data size of 10000.

| Correctness | Size | Model  |
|-------------| -----| -----  |
| ece408-high | 10000 (default) | 0.8562 |
| ece408-low  | 10000 (default) | 0.629  |

Use 

    rai -p <project folder> --submit

to mark your submission. This will notify the teaching staff of which `report.pdf` draft to consider.

## Milestone 3
**A New GPU Convolution Layer in MxNet: Due Friday December 1st, 2017**

A draft of the `report.pdf` with content up through Milestone 3 must be submitted **through rai** for this milestone.

### 3.1 Add a simple GPU forward implementation

**Goal: successfully edit code and run in rai**

Modify `ece408_src/new-forward.cuh` to implement a forward GPU convolution.
You may run your code with `python m3.1.py`. It takes the same arguments as `m2.1py`.

### 3.2 Create a profile with `nvprof`.

Once you have a simple GPU implementation, modify `rai_build.py` to create a profile with NVPROF.
You should see something like this:

    ✱ Running nvprof python /src/m3.1.py
    Loading fashion-mnist data... done
    ==308== NVPROF is profiling process 308, command: python /src/m3.1.py
    Loading model... done
    Time: 14.895404
    Correctness: 0.8562 Batch Size: 10000 Model: ece408-high
    ==308== Profiling application: python /src/m3.1.py
    ==308== Profiling result:
    Time(%)      Time     Calls       Avg       Min       Max  Name
    99.43%  14.8952s         1  14.8952s  14.8952s  14.8952s  void mxnet::op::forward_kernel<mshadow::gpu, float>(float*, mxnet::op::forward_kernel<mshadow::gpu, float> const *, mxnet::op::forward_kernel<mshadow::gpu, float> const , int, int, int, int, int, int)

In this example, the forward layer took 14.8954 seconds, and the forward_kernel took 14.8952 seconds.

Again, use `rai -p <project folder> --submit` to submit your code.

## Final Submission
**An Optimized Layer and Final Report: Due Friday December 15th, 2017**

### Optimized Layer

Optimize your GPU convolution.

Your implementation will be partially graded on its performance relative to other optimized implementations from the class.

All of your code for this and the later milestones must be executed between `auto start = ...` and `auto end = ...` in `new-inl.h`.
The easiest way to ensure this is that all of your code should be in `forward()` or called by `forward()` from `new-forward.cuh` or `new-forward.h`.
Do not modify any timing-related code.

You may use nvprof to collect more detailed information through timeline and analysis files.

    nvprof -o timeline.nvprof <your executable>
    nvprof --analysis-metrics -o analysis.nvprof <your executable>

you can collect the generated files by following the download link reported by rai at the end of the execution.
`--analysis-metrics` significantly slows the run time, you may wish to modify the python scripts to run on smaller datasets during this profiling.

**Only your last `--submit` will be graded. Be sure that your final `--submit` is the one you want to be graded.**

### Final Report

You should provide a brief PDF final report `report.pdf`, with the following content.

1. **Baseline Results**
    1. M1.1: mxnet CPU layer performance results (time)
    2. M1.2: mxnet GPU layer performance results (time, `nvprof` profile)
    3. M2.1: your baseline cpu implementation performance results (time)
    4. M3.1: your baseline gpu implementation performance results (time, `nvprof` profile)
2. **Optimization Approach and Results**
    * how you identified the optimization opportunity
    * why you thought the approach would be fruitful
    * the effect of the optimization. was it fruitful, and why or why not. Use nvprof as needed to justify your explanation.
    * Any external references used during identification or development of the optimization
3. **References** (as needed)

### Final submission through RAI

To make an official project submission, you will run

    rai -p . --submit

The `--submit` flag
* enforces a specific rai_build.yml
* requires the existence of `report.pdf`
* Records your batch size, operation time, user time, system time, and correctness in a database. An anonymous version of these results (not your code!) will be visible to other students.

The submit flag will run `build.sh`, which should build your code and install the python bindings (like the provided `build.sh`). Then it will run `python final.py`. You must ensure that your project works under those constraints. Do not modify `final.py`.

**Only your most recent submission will be graded. Ensure that your final submission is the one you want to be graded**.

### Rubric

1. Optimized Layer (50%)
    * correctness (25%)
    * relative ranking (25%)
2. Final Report ( 50% )
    * Milestone 2 ( 5% )
    * Milestone 3 ( 10% )
    * Final Submission ( 35% )

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

The `x`, `y`, and `k` tensors constructed in `new-inl.h`/`Forward()` have the following data layout:

| Tensor | Descrption | Data Layout |
| -- | -- | -- |
| `x` | Input data     | batch size * input channels * y * x |
| `y` | Output data    | batch size * output channels * y * x |
| `k` | kernel weights | output channels * input channels * y * x |

You can see this being constructed in `new-inl.h`/`InferShape()`.

## Extras

### Provided Model Weights

The execution environment provides two models for the new convolutional layer you implement:

| Prefix | Test Set Accuracy |
| -- | -- |
| `models/ece408-high` | 0.8562 |
| `models/ece408-low` | 0.6290 |

When testing your implementation, you should achieve these accuracy values for the CPU or GPU implementation.

There is also one model used in milestone 1.


| Prefix | Test Set Accuracy |
| -- | -- |
| `models/baseline` | 0.8673 |

### Checking for Errors

Within MXNet, you can use `MSHADOW_CUDA_CALL(...);` as is done in `new-forward.cuh`.
Or, you can define a macro/function similar to `wbCheck` used in WebGPU.

### Comparing GPU implementation to CPU implementation

It may be hard to directly debug by inspecting values during the forward pass since the weights are already trained and the input data is from a real dataset.
You can always extract your implementations into a separate set of files, generate your own test data, and modify `rai_build.yml` to build execute your separate test code instead of the MXNet code while developing.

A simple code is provided in `build_example`. You could modify the `build` step of rai_build.yml in the following way to compile and run it:

    commands:
        build:
            - echo "Building arbitrary code"
            - make -C /src/build_example
            - echo "Running compiled code"
            - /src/build_example/main

### Offline Development

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