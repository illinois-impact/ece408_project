# ECE408/CS483 Final Project

## Introduction

This is the skeleton code for the Spring 2018 ECE408 / CS483 course project.
In this project, you will:

* Get practical experience by using, profiling, and modifying MXNet, a standard open-source neural-network framework.
* Demonstrate command of CUDA and optimization approaches by designing and implementing an optimized neural-network convolution layer forward pass.

The project will be broken up into 3 milestones and a final submission. Read the description of the final report before starting, so you can collect the necessary info along the way.
Each milestone will consist of an updated report (culminating in the final report).

You will be working in teams of 3.

You are expected to adhere to University of Illinois academic integrity standards.
Do not attempt to subvert and of the performance-measurement aspects of the final project.
If you are unsure about whether something does not meet those guidelines, ask a member of the teaching staff.

## Table of Contents

* [Milestone 1: Due 3/7@5pm](#milestone-1)
* [Milestone 2: Due 3/21@5pm](#milestone-2)
* [Milestone 3: Due 4/18@5pm](#milestone-3)
* [Milestone 4: Due 5/2@5pm](#milestone-4)
* [Final Submission: Due 5/11@5pm](#final-submission)
* [Rubric](#rubric)
* [Final Report](#final-report)
* [Extras](#extras)

## Milestone 1

Due March 7 @ 5pm

As with all milestones, you will include an updated PDF `report.pdf` in the project directory you submit with rai.
This report should contain all of the deliverables.
This report should contain your names, netids, rai ids (if different), team names, and school affiliation (Chicago or UIUC).

| Deliverables |
| ------------ |
| Register your team in the google sheet. |
| Report: Include a list of all kernels that collectively consume more than 90% of the program time. |
| Report: Include a list of all CUDA API calls that collectively consume more than 90% of the program time. |
| Report: Include an explanation of the difference between kernels and API calls |
| Report: Show output of rai running MXNet on the CPU |
| Report: List program run time |
| Report: Show output of rai running MXNet on the GPU |
| Report: List program run time |
| Use `rai -p <project folder> --submit=m1` to mark your job for grading |

You and your team should agree on a team name and enter it in this [google sheet](https://goo.gl/forms/Sz0fLOWscPP1hsS12).

Clone this repository to get the project folder.

    git clone https://github.com/illinois-impact/ece408_project.git

Download the rai binary for your platform. 
You will probably use it for development, and definitely use it for submission.


| Operating System | Architecture | Stable Version (0.2.20) Link                                                             |
| ---------------- | ------------ | ------------------------------------------------------------------------------- |
| Linux            | amd64        | [URL](https://github.com/rai-project/rai/releases/download/v0.2.23/linux-amd64.tar.gz)   |
| OSX/Darwin       | amd64        | [URL](https://github.com/rai-project/rai/releases/download/v0.2.23/darwin-amd64.tar.gz)  |
| Windows          | amd64        | [URL](https://github.com/rai-project/rai/releases/download/v0.2.23/windows-amd64.tar.gz) |

You should have received a `.rai_profile` file by email.
Put that file in `~/.rai_profile` (Linux/macOS) or `%HOME%/.rai_profile` (Windows).
Your `.rai_profile` should look something like this (indented with tabs!)

    profile:
        firstname: <your-given-name>
        lastname: <your-surname>
        username: <your-username>
        email: <your-institution-email>
        access_key: <your-access-key>
        secret_key: <your-secret-key>
        affiliation: uiuc

You will need to add your team name in the following way:

    profile:
        firstname: <your-given-name>
        lastname: <your-surname>
        username: <your-username>
        email: <your-institution-email>
        access_key: <your-access-key>
        secret_key: <your-secret-key>
        affiliation: uiuc
        team:
            name: <your-team-name>

Some more info is available on the [Client Documentation Page](https://github.com/rai-project/rai).

Run the built-in MXNet forward pass using rai

Consult `m1.1py` to examine the neural-network architecture used in this project.

Use RAI to run a batch forward pass on some test data.

    rai -p <project-folder>

This will upload your project directory to rai (running on AWS) and move it to `/src`, where the execution specified in `rai_build.yml` will occur. 

The `image:` key specifies the environment that the rest of the execution will occur in.
This environment includes a prebuilt MXNet (so rai will only do a partial compile with your code) as well as the model definition and the training data.

The `resources:` key specifies what computation resources will be available to the execution.

The `commands:` key specifies the recipe that rai will execute. First, the project files are copied to the `/build` directory.
Then the files in `ece408_src` are copied to `src/operator/custom/` in the MXNet source tree.
MXNet is recompiled, and the Python bindings are installed.
`python /src/m1.1.py` runs the `m1.1.py` python program.

You should see the following output:

    Loading fashion-mnist data... done
    Loading model... done
    EvalMetric: {'accuracy': 0.8444}

Modify `rai_build.yml` to use `/usr/bin/time` to measure the elapsed time of the whole program.

    - /usr/bin/time python m1.1.py

Next, we will run on the GPU!

Compare `m1.2.py` and `m1.1.py`. You'll see that it is the same, except for `mx.gpu()` has been substituted for `mx.cpu()`. This is how we tell MXNet that we wish to use a GPU instead of a CPU.

Modify `rai_build.yml` to time `python m1.2.py`

Again, submit the job to rai

    rai -p <project-folder>

Next, we will learn how to use `nvprof` to profile the execution

Once you've gotten the appropriate accuracy results, generate a profile using nvprof. You will be able to use nvprof to evaluate how effective your optimizations are.
As described above, make sure `rai_build.yml` is configured for a GPU run.
Then, modify `rai_build.yml` to generate a profile instead of just execuing the code.

    nvprof python m1.2.py

You should see something that looks like the following:

~~~bash 
==15163== NVPROF is profiling process 15163, command: python m1.2.py
Loading model...[13:14:46] src/operator/././cudnn_algoreg-inl.h:112: Running performance tests to find the best convolution algorithm,this can take a while... (setting env variable MXNET_CUDNN_AUTOTUNE_DEFAULT to 0 to disable)
 done
EvalMetric: {'accuracy': 0.8444}
==15163== Profiling application: python m1.2.py
==15163== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   39.66%  88.817ms      1002  88.639us  67.553us  112.64us  maxwell_scudnn_128x32_relu_interior_nn
                   30.78%  68.932ms      1000  68.932us  6.4010us  137.51us  sgemm_32x32x32_NT_vec
                    7.08%  15.849ms      1018  15.569us     608ns  1.0653ms  [CUDA memcpy HtoD]
                    6.58%  14.726ms      1000  14.725us  3.0080us  30.145us  void 

...

      API calls:   38.57%  1.68099s        22  76.409ms  20.706us  844.45ms  cudaStreamCreateWithFlags
                   29.08%  1.26736s        50  25.347ms     560ns  328.56ms  cudaFree
                   20.76%  904.87ms        27  33.514ms  45.260us  902.77ms  cudaMemGetInfo
                    4.00%  174.45ms      4520  38.595us  1.9200us  1.2852ms  cudaStreamSynchronize
                    3.03%  131.95ms      1506  87.617us  11.564us  1.3006ms  cudaMemcpy2DAsync

...
~~~

The GPU Activities section shows the kernels and memory transfers, and the API calls section shows the CUDA API calls that are executed.
There are columns corresponding to percentage of time consumed, total time, number of calls, and average/min/max time of those calls.
Think about the distinction between a CUDA API call and a kernel launch, and describe it briefly in your report.
The CUDA documentation describes [kernels](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#kernels) and the [programming interface](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-interface).

You can find more information about nvprof in the [CUDA Toolkit Documentation](http://docs.nvidia.com/cuda/profiler-users-guide/index.html#nvprof-overview)

Use 

    rai -p <project folder> --submit=m1

to mark your submission. This will notify the teaching staff of which `report.pdf` draft to consider.

## Milestone 2

Due March 21 @ 5pm

As with all milestones, you will include an updated PDF `report.pdf` with all of the required deliverables for this and preceeding milestones.

| Deliverables |
| ------------ |
| Everything from Milestone 1 |
| Create a  CPU implementation |
| Report: List full program run time |
| Report: List Op Times |
| Use `rai -p <project folder> --submit=m2` to mark your job for grading |

See the [description](#markdown-header-skeleton-code-description) of the skeleton code for background information, including the data storage layout of the tensors.

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

Because this operator is different than the built-in MXNet operator, you will need to load a different model.
`m2.1.py` handles this for you.
Modify `rai_build.yml` to invoke

    python m2.1.py

When your implementation is correct, you should see output like this:

    Loading fashion-mnist data... done
    Loading model... done
    Op Time: [ some time ]
    Op Time: [ some time ]
    Correctness: 0.8451 Model: ece408

Every time your layer is invoked, it will print the "Op Time," the time spent working on that layer.
Since the network has two convolutional layers, two times will be printed.
You can time the whole program execution by modifying `rai_build.yml` with

    /usr/bin/time python m2.1.py

`m2.1.py` takes one optional argument: the dataset size.
If the correctness for each possible model is as below, you can be reasonably confident your implementation is right.
The correctness does depend on the data size. Check your correctness on the full data size of 10000.

For example, you could modify `rai_build.yml` to run

    python m2.1.py 100

| Model | Number of Images | Correctness  |
|-------------| -----| -----  |
| ece408 | 10000 (default) | 0.8451 |
| ece408 | 100 | 0.88 |
| ece408 | 10 | 1 |

The provided `m2.1.py` is identical to the one used by `--submit=m2`.
You may modify `m2.1.py` as you please, but check that `--submit=m2` will still invoke your code correctly.

Use

    rai -p <project folder> --submit=m2

to mark your submission.

## Milestone 3

Due April 18 @ 5pm

| Deliverables |
| ------------ |
| Everything from Milestone 2 |
| Implement a GPU Convolution |
| Report: demonstrate `nvprof` profiling the execution |
| Use `rai -p <project folder> --submit=m3` to mark your job for grading |

### Create a GPU Implementation

Modify `ece408_src/new-forward.cuh` to create GPU implementation of the forward convolution.

Modify `rai_build.yml` to run

    python m3.1.py

to use your GPU implementation.
When it is correct, it will show the same correctness as Milestone 2.

### Use `nvprof` and NVVP for initial Performance Results

Modify `rai_build.yml` to use nvprof to save some timeline and analysis information, as described in [nvprof](#profiling).
Use the NVIDIA Visual Profiler and your analysis information to describe the effect that your optimizations had on the performance of your convolution.
If possible, you should try to separate the effect of each optimization in your analysis.
The [NVVP on EWS](#nvvp-on-ews) section describes how to install NVVP.

Use

    rai -p <project folder> --submit=m3

to mark your submission.

## Milestone 4

Due May 2 @ 5pm

| Deliverables |
| ------------ |
| Everything from Milestone 3 |
| Implement three GPU optimizations |
| Report: Describe the optimization |
| Report: demonstrate `nvprof` profiling the execution |
| Report: use NVVP to analyze your optimization |
| Use `rai -p <project folder> --submit=m4` to mark your job for grading |

### 3.1 Add three GPU Optimization

For this milestone, you should attempt at least three GPU optimizations.

Describe the optimizations in your `report.pdf`.

### 3.2 Performance Analysis with `nvprof` and NVVP

Analyze the effect of each optimization using the same techniques as Milestone 3.

Use 
    
    rai -p <project folder> --submit=m4
    
to submit your project folder.

## Final Submission

Due May 11 @ 5pm

| Deliverables |
| ------------ |
| Everything from Milestone 4 |
| Implement final GPU optimizations |
| Report: Describe and analyze the optimizations |
| Report: demonstrate `nvprof` profiling the execution |
| Use `rai -p <project folder> --submit=final` to mark your job for grading |

### Optimized Layer

Optimize your GPU convolution.

Your implementation must work with `rai -p <project-folder> --submit=final`.
This means all your source files must be in `ece408_src`, and your implementation must work when they are copied to `src/operator/custom` in the MXNet tree, and `make` is invoked on the MXNet tree.
This is done in the provided `rai_build.yml`.
Likewise, the provided `final.py` provides an example of the script that will be used to time your implementation.

All of your code for this and the later milestones must be executed between `auto start = ...` and `auto end = ...` in `new-inl.h`.
The easiest way to ensure this is that all of your code should be in `forward()` or called by `forward()` from `new-forward.cuh` or `new-forward.h`.
Do not modify any timing-related code.

Use `rai -p <project folder> --submit=final` to submit your project folder.

### Final Report

You've been building this final report through all the milestones.
Keep the content from the earlier milestones, but be sure to include the following:

* Your team name
* Your team member names
* your netids
* your UINs

The final report should include at least the following information for each optimization

1. **Optimization Approach and Results**
    * how you identified the optimization opportunity
    * why you thought the approach would be fruitful
    * the effect of the optimization. was it fruitful, and why or why not. Use nvprof and NVVP to justify your explanation.
    * Any external references used during identification or development of the optimization
    * How  your team organized and divided up this work.
2. **References** (as needed)
3. **(Optional) Suggestions for Improving Next Year**

### Rubric

The overall project score will be computed as follows:

1. Milestone 1 ( 5% )
2. Milestone 2 ( 10% )
3. Milestone 3 ( 10% )
    * Optimization 1
4. Final Optimizations ( 60% )
    * Optimization 2 ( 10% )
    * Optimization 3 ( 10% )
    * Optimization 4 ( 10% )
    * Optimization 5 ( 10% )
    * Optimization 6 ( 10% )
    * Additional Optimizations / detailed insights ( 10% )
5. Performance Ranking ( 10% )
6. Report Style (10 %)
    * Clear, concise writing, good layout, and good organization will be rewarded.

Each optimization will be graded as follows:

1. Explanation of Performance Impact ( 40% )
2. Correctness ( 60% )

The Performance Ranking will be graded as follows:

1. The median performance will be determined (how well the class did as a whole)
2. Your performance will be converted to a number of standard deviations above/below that median (how well you did compared to the class).
3. That value will be linearly mapped into the space of 0-10 to determine the ranking grade.

The ranking is determined by the total run time of the two layer invocations.
If your implementation is not correct, you will get a 0 for this component of the grade.
The `rai ranking` command is not the final word: the staff will re-run all final submissions multiple times and choose the fastest result as your time.
THe ranking is determined solely by the values printed by `Op Time:` during your run.
That `Op Time` is computed by wrapping the MXNet op that you implement in a timer.

## Extras

### Checking for Errors

Within MXNet, you can use `MSHADOW_CUDA_CALL(...);` as is done in `new-forward.cuh`.
Or, you can define a macro/function similar to `wbCheck` used in WebGPU.

### Profiling

You can gather detailed GPU profile information with `nvprof` and view that information with `nvvp`.

You can see some simple information like so (as we did in milestone 1):

    nvprof <your command here>

You can gather a timeline file like the following:

    nvprof -o timeline.nvprof <your command here>

This will generate timeline.nvprof.

You can additionally gather some detailed performance metrics.

    nvprof -o timeline.nvprof <your command here>
    nvprof --analysis-metrics -o analysis.nvprof <the same command>

This will generate `timeline.nvprof` and `analysis.nvprof`.
`--analysis-metrics` significantly slows the run time, you may wish to modify the python scripts to run on smaller datasets during this profiling.

You will need to follow the link rai prints after the execution to retrieve these files.
You can use the NVIDIA Visual Profiler (nvvp) to import those files.
You will need to install nvvp on your own machine. It can be downloaded as part of the CUDA SDK.

To import the files:
* File > import > select nvprof > next > single process > next
* timeline data file should be your timeline.nvprof
* event/metrics data file should be your analysis.nvprof.
* finish

### NVVP on EWS

The process will be similar for any machine without an NVIDIA GPU (like your linux laptop).

If you wish to install it on Windows or macOS, the CUDA Toolkit installer may partially fail if you do not have an NVIDIA GPU.
The teaching staff doesn't support this, but you may be able to figure it out.

Establish an ssh session with x-forwarding

    ssh -Y <netid>@linux.ews.illinois.edu

Download CUDA toolkit for CentOS 7 and install to `~/software/cuda-9.0` (You may choose a different location).
This takes a while (1GB+ download and install).

    mkdir -p $HOME/software \
    && wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda_9.0.176_384.81_linux-run -O cuda9.run \
    && chmod +x cuda9.run \
    && ./cuda9.run --silent --toolkit --toolkitpath=$HOME/software/cuda-9.0

Free up your EWS space (I'm not sure what the disk quotas are)

    rm cuda9.run

Optional: modify .bashrc to add `~/software/cuda-9.0/bin` to your path. Or, just run it directly

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

If you'd like to develop using a local copy of MXNet, you may do so. Keep in mind your project will be evaluated through rai. Your submission must work through rai.

The MXNet instructions are available [here](https://mxnet.incubator.apache.org/get_started/install.html). A short form of them follows for Ubuntu.

    # install some prereqs
    sudo apt install -y build-essential git libopenblas-dev liblapack-dev libopencv-dev python-pip python-dev python-setuptools python-numpy
    # download MXNet release 0.11.0
    git clone git@github.com:apache/incubator-mxnet.git --recursive --branch 0.11.0
    # build MXNet
    nice -n20 make -C incubator-mxnet -j$(nrpoc) USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1 USE_BLAS=openblas
    # install python bindings
    pip install --user -e incubator-mxnet/python

You can always uninstall the python package with

    pip uninstall mxnet

Download the fashion-mnist dataset and generate the larger data (this may take a few minutes and consume a few hundred megabytes of disk space).

    python generate-data.py

Modify the python forward convolution scripts to point to where you downloaded fashion-mnist

    ... load_mnist(path="fashion-mnist", ...)

Download the trained models (for the existing MXNet implementation and your implementation) using 

    mkdir -p models \
    && wget -P models \
        https://github.com/illinois-impact/ece408_mxnet_docker/raw/master/models/baseline-0002.params \
        https://github.com/illinois-impact/ece408_mxnet_docker/raw/master/models/baseline-symbol.json \
        https://github.com/illinois-impact/ece408_mxnet_docker/raw/master/models/ece408-002.params \
        https://github.com/illinois-impact/ece408_mxnet_docker/raw/master/models/ece408-symbol.json
        
Modify the python forward convolution scripts to point to where you downloaded fashion-mnist

    lenet_model = mx.mod.Module.load( prefix='models/baseline' ...

Build your modified MXNet

    cp <your source files> incubator-mxnet/src/operator/custom
    make -C incubator-mxnet/


### Skeleton Code Description

`new-forward.h` and `new-forward.cuh` contain skeleton implementations for CPU and GPU convolutions. You can complete the project by modifying only these two files. These functions are called from `Forward()` in `new-inl.h`.

The code in `new-inl.h`, `new.cc`, and `new.cu` describes the convolution layer to MXNet. You should not modify these files. They are provided for your curiosity.
As of rai 0.2.20, When you use the `--submit` flag, a golden version of these files from [here](https://github.com/cwpearson/2017fa_ece408_mxnet_docker/tree/master/ece408-src) is used.

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


## License

NCSA/UIUC © 2018 [Carl Pearson](https://cwpearson.github.io)
