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

### Using RAI

    rai -p <project-folder>

This causes the following things to happen:

* RAI client: upload the project folder to AWS
* RAI client: Notify a RAI server that a job is ready
* RAI server: downloads the folder from AWS
* RAI server: starts the docker container specified in `rai_build.yml`
* RAI server: uses docker container to execute the steps specified in `rai_build.yml`
* RAI server: uploads `/build` directory to AWS
* RAI client: gives you the link to that build directory

So, if you want any results from your run, you need to generate those results in `/build`. 
The provided `rai_build.yml` moves everything to the `/build` directory in an early step.

### Final Submissions through RAI

To make an official project submission, you will run

    rai -p <project folder> --submit=<submission kind>

The `--submit` flag accepts `m2` for milestone 2, `m3` for milestone 3, and `final` for the final submission. 

    rai -p <project-folder-with-working-cpu-implementation> --submit=m2

Using the `--submit` flag
* enforces a specific `rai_build.yml` depending on which kind of submission you do.
* requires the existence of `report.pdf`
* Records your operation time, user time, system time, and correctness in a database. An anonymous version of these results (not your code!) will be visible to other students.

To ensure that `--submit` goes smoothly, ensure your code works with the provided python scripts. They are similar to the ones used by `--submit`.

**Only your most recent submission will be graded. Ensure that your final submission is the one you want to be graded**.

## Milestone 1

**Getting Started: Due Friday November 10th, 2017**

Nothing must be turned in for this milestone, but content will be used in Milestone 2.

### Getting Set Up and Getting Bugfixes

Clone this repository to get the project directory.

    git clone https://github.com/webgpu/2017fa_ece408_project.git

Download the rai binary for your platform. You will probably use it for development, and definitely use it for submission.

| Operating System | Architecture | Stable Version (0.2.18) Link                                                             |
| ---------------- | ------------ | ------------------------------------------------------------------------------- |
| Linux            | amd64        | [URL](https://github.com/rai-project/rai/releases/download/v0.2.18/linux-amd64.tar.gz)   |
| OSX/Darwin       | amd64        | [URL](https://github.com/rai-project/rai/releases/download/v0.2.18/darwin-amd64.tar.gz)  |
| Windows          | amd64        | [URL](https://github.com/rai-project/rai/releases/download/v0.2.18/windows-amd64.tar.gz) |

You should have received a `.rai_profile` file by email.
Put that file in `~/.rai_profile` (Linux/macOS) or `%HOME%/.rai_profile` (Windows).
As soon as you and your two teammates agree on a team name, fill in the corresponding entry in your `.rai_profile`.
Your `.rai_profile` should look something like this (indented with tabs!)

    profile:
        firstname: <your-given-name>
        lastname: <your-surname>
        username: <your-username>
        email: <your-access-key>
        access_key: <your-access-key>
        secret_key: <your-secret-key>
        affiliation: uiuc
        team:
            name: <your-team-name-here>

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

The `commands:` key specifies the recipe that rai will execute. First, the project files are copied to the `/build` directory.
Then the files in `ece408_src` are copied to `src/operator/custom/` in the MXNet source tree.
MxNet is recompiled, and the pythong bindings are installed.
`python /src/m1.1.py` runs the `m1.1.py` python program.

You should see the following output:

    Loading fashion-mnist data... done
    Loading model... done
    EvalMetric: {'accuracy': 0.8673}

**Deliverables (to be submitted with Milestone 2)** 
In your report, confirm that this is the output you see. Use `/usr/bin/time` to measure the elapsed time of the whole python program.

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

**Deliverables (to be submitted with Milestone 2)** 
In your report, confirm the accuracy. Use `/usr/bin/time` to measure the elapsed time of the whole python program.

### 1.3 Generate a NVPROF Profile

**Goal: Be able to use nvprof for performance evaluation**

Once you've gotten the appropriate accuracy results, generate a profile using nvprof. You will be able to use nvprof to evaluate how effective your optimizations are.
As described above, make sure `rai_build.yml` is configured for a GPU run.
Then, modify `rai_build.yml` to generate a profile instead of just execuing the code.

    nvprof python m1.2.py

You should see something that looks like the following:

    ✱ Running nvprof python m1.2.py
    Loading fashion-mnist data... done
    ==308== NVPROF is profiling process 308, command: python     /src/m1.2.py
    Loading model... done
    EvalMetric: {'accuracy': 0.8673}
    ==308== Profiling application: python /src/m1.2.py
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
Each line correspnds to a CUDA kernel or an API call.
There are columns corresponding to percentage of time consumed, total time, number of calls, and average/min/max time of those calls.

You can find more information about nvprof in the [CUDA Toolkit Documentation](http://docs.nvidia.com/cuda/profiler-users-guide/index.html#nvprof-overview)

**Deliverables (to be submitted with Milestone 2)** 
In your report, list a table of the most time-consuming kernels.

## Milestone 2
**A New CPU Convolution Layer in MxNet: Due Friday November 17th, 2017**

A draft of the `report.pdf` with content up through Milestone 2 must be submitted **through rai** for this milestone.

See the [description](#markdown-header-skeleton-code-description) of the skeleton code for background information, including the data storage layout of the tensors.

### 2.1 Add a simple CPU forward implementation

**Goal: successfully edit code and run in rai**

Modify `ece408_src/new-forward.h` to implement the forward convolution described in [Chapter 16 of the textbook](https://wiki.illinois.edu/wiki/display/ECE408Fall2017/Textbook+Chapters).
The performance of the CPU convolution is not part of the project evaluation.
The algorithm is also below, for your convenience

    for b = 0 .. B)                    // for each image in the batch 
        for m = 0 .. M                 // for each output feature maps
            for h = 0 .. H_out         // for each output element
                for w = 0 .. W_out
                {
                    y[b][m][h][w] = 0;
                    for c = 0 .. C     // sum over all input feature maps
                        for p = 0 .. K // KxK filter
                            for q = 0 .. K
                                y[b][m][h][w] += x[b][c][h + p][w + q] * k[m][c][p][q]
                }

Unlike the convolutions described in the class, note that this one is not centered on the input image.

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

For example, you could modify `rai_build.yml` to run

    - python m2.1.py ece408-low 100

| Correctness | Size | Model  |
|-------------| -----| -----  |
| ece408-high | 10000 (default) | 0.8562 |
| ece408-low  | 10000 (default) | 0.629  |

The provided `m2.1.py` is identical to the one used by `--submit=m2`.
You may modify `m2.1.py` as you please, but check that `--submit=m2` will still invoke your code correctly.

**Deliverables**
Use 

    rai -p <project folder> --submit=m2

to mark your submission. This will notify the teaching staff of which `report.pdf` draft to consider.

This will run your code against the two datasets, and check the time and correctness.

Your `report.pdf` at this stage should contain content up through M2.1  described in the final report section.


## Milestone 3
**A New GPU Convolution Layer in MxNet: Due Friday December 1st, 2017**

A draft of the `report.pdf` with content up through Milestone 3 must be submitted **through rai** for this milestone.

### 3.1 Add a simple GPU forward implementation

**Goal: successfully edit code and run in rai**

Modify `ece408_src/new-forward.cuh` to implement a forward GPU convolution.
You may run your code with `python m3.1.py`. It takes the same arguments as `m2.1py`.
Again, if you choose to modify `m3.1.py`, be sure the original still works with your convolution implementation.

### 3.2 Create a GPU profile with `nvprof`.

Once you have a simple GPU implementation, modify `rai_build.py` to create a profile with NVPROF.
You should see something like this:

    ✱ Running nvprof python m3.1.py
    Loading fashion-mnist data... done
    ==308== NVPROF is profiling process 308, command: python m3.1.py
    Loading model... done
    Time: 14.895404
    Correctness: 0.8562 Batch Size: 10000 Model: ece408-high
    ==308== Profiling application: python /src/m3.1.py
    ==308== Profiling result:
    Time(%)      Time     Calls       Avg       Min       Max  Name
    99.43%  14.8952s         1  14.8952s  14.8952s  14.8952s  void mxnet::op::forward_kernel<mshadow::gpu, float>(float*, mxnet::op::forward_kernel<mshadow::gpu, float> const *, mxnet::op::forward_kernel<mshadow::gpu, float> const , int, int, int, int, int, int)

In this example, the forward layer took 14.8954 seconds, and the forward_kernel took 14.8952 seconds.

**Deliverables**
Again, use `rai -p <project folder> --submit=m3` to submit your code.

Your `report.pdf` at this stage should contain content up through M3.1 described in the final report section.

## Final Submission
**An Optimized Layer and Final Report: Due Friday December 15th, 2017**

### Optimized Layer

Optimize your GPU convolution.

Your implementation will be partially graded on its performance relative to other optimized implementations from the class.

Your implementation must work with `rai -p <project-folder> --submit=final`.
This means all your source files must be in `ece408_src`, and your implementation must work when they are copied to `src/operator/custom` in the mxnet tree, and `make` is invoked on the mxnet tree.
This is done in the provided `rai_build.yml`.
Likewise, the provided `final.py` provides an example of the script that will be used to time your implementation.

All of your code for this and the later milestones must be executed between `auto start = ...` and `auto end = ...` in `new-inl.h`.
The easiest way to ensure this is that all of your code should be in `forward()` or called by `forward()` from `new-forward.cuh` or `new-forward.h`.
Do not modify any timing-related code.

You may use nvprof to collect more detailed information through timeline and analysis files.

    nvprof -o timeline.nvprof <your executable>
    nvprof --analysis-metrics -o analysis.nvprof <your executable>

you can collect the generated files by following the download link reported by rai at the end of the execution.
`--analysis-metrics` significantly slows the run time, you may wish to modify the python scripts to run on smaller datasets during this profiling.

**Deliverables**

### Final Report

You should provide a brief PDF final report `report.pdf`, with the following content.
The report does not need to be a particular length, but should be long enough to cover all of this content.

1. **Baseline Results**
    1. M1.1: mxnet CPU layer correctness and elapsed time for the whole python program.
     You can measure the elapsed time of the program with the `time` command.
    2. M1.2/M1.3: mxnet GPU layer performance results (`nvprof` profile). Include your profile, and describe in a few words how the GPU is spending its time.
    This is to confirm you can generate a profile and can interpret it.
    3. M2.1: your baseline cpu implementation correctness and performance results (time).
    The `Op Time:` printed by the program will show the time just for the convolution layer.
    The implementation should have the expected correctness.
    4. M3.1: your baseline gpu implementation performance results (time, `nvprof` profile).
    The implementation should have the expected correctness.
2. **Optimization Approach and Results**
    * how you identified the optimization opportunity
    * why you thought the approach would be fruitful
    * the effect of the optimization. was it fruitful, and why or why not. Use nvprof as needed to justify your explanation.
    * Any external references used during identification or development of the optimization
3. **References** (as needed)
4. **(Optional) Suggestions for Improving Next Year**

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

### Profiling

You can gather detailed GPU profile information with `nvprof`.
To use `nvprof`, you'll need to be using the `cwpearson/2017fa_ece408_mxnet_docker:amd64-gpu-latest` image.

You can see some simple information like so (as we did in milestone 1):

    nvprof <your command here>

You can gather a timeline file like the following:

    nvprof -o timeline.nvprof <your command here>

This will generate timeline.nvprof.

You can additionally gather some detailed performance metrics.

    nvprof -o timeline.nvprof <your command here>
    nvprof --analysis-metrics -o analysis.nvprof <the same command>

This will generate `timeline.nvprof` and `analysis.nvprof`.

You will need to follow the link rai prints after the execution to retrieve these files.
You can use the NVIDIA Visual Profiler (nvvp) to import those files.
You will need to install nvvp on your own machine. It can be downloaded as part of the CUDA SDK.

To import the files:
* File > import > select nvprof > next > single process > next
* timeline data file should be your timeline.nvprof
* event/metrics data file should be your analysis.nvprof.
* finish

### Installing NVVP on EWS

This will install nvvp on the EWS machines. The process will be similar for any machine without an NVIDIA GPU.

Establish an ssh session with x-forwarding

    ssh -Y <netid>@linux.ews.illinois.edu

Download CUDA toolkit for CentOS 7

    wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda_9.0.176_384.81_linux-run -O cuda9.run

Install nvvp (to `~/software/cuda-9.0`. You may choose a different location.) This takes a while.

    chmod +x cuda9.run
    ./cuda9.run --silent --toolkit --toolkitpath=$HOME/software/cuda-9.0

Free up your EWS space (I'm not sure what the disk quotas are)

    rm cuda9.run

Optional: modify .bashrc to add `~/software/cuda-9.0/bin` to your path. Or, just run it

    ~/software/cuda-9.0/bin/nvvp &

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