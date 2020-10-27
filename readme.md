# ECE408/CS483 Final Project

## Introduction

This is the skeleton code for the Fall 2020 ECE408 / CS483 / CSE408 course project.

In this final project, you will be implementing and optimizing the forward-pass of a convolutional layer using CUDA. Convolutional layers are the primary building blocks of convolutional neural networks (CNNs), which are used in many machine learning tasks like image classification, object detection, natural language processing, and recommendation systems. In general, CNNs work well on tasks where the data/input features have some level of spatial relationship.

You will be working with a **modified** version of the LeNet-5 architecture shown below.

![LenetImage](https://lh5.googleusercontent.com/84RlneM7JSDYDirUr_ceplL4G3-Peyq5dkLJTe2f-3Bj9KuWZjsH2A9Qq5PO5BRLrVfWGPnI3eQu8RkTPgyeUf9ZOWY9JbptVJy9LceAyHRn-O0kbzprx88yb82a5dnCR7EDP7n0)

*Source: http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf*

Your optimized CUDA implementation of the convolutional layer will be used to perform inference for layers C1 and C3 (shown in red) in the figure above. We will be leveraging the [mini-dnn-cpp](https://github.com/iamhankai/mini-dnn-cpp) (Mini-DNN) framework for implementing the modified LeNet-5. 

We will be using the [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist), where the inputs to the network will be a batch of 10,000 single channel images, each with dimensions of 86 x 86 pixels. The output layer consists of 10 nodes, where each node represents the likelihood of the input belonging to one of the 10 classes (T-shirt, dress, sneaker, boot etc.)

The overall learning objectives for this project are:
* Demonstrating command of CUDA and optimization approaches by designing and implementing an optimized neural-network convolutional layer forward pass
* Obtaining practical experience in analyzing and fine tuning CUDA kernels through the use of profiling tools like Nsight Systems (`nsys`) and Nsight-Compute (`nv-nsight-cu`)

You will be working in teams of 3 (no exceptions unless approved by the course staff).

*You are expected to adhere to University of Illinois academic integrity standards. Do not attempt to subvert any of the performance-measurement aspects of the final project. If you are unsure about whether something does not meet those guidelines, ask a member of the teaching staff.*

## Table of Contents

* [Milestone 1: Team Registration](#milestone-1-team-registration)
* [Milestone 2: Rai installation, CPU convolution](#milestone-2-rai-installation-cpu-convolution)
* [Milestone 3: Baseline Convolutional Kernel](#milestone-3-baseline-convolutional-kernel)
* [Milestone 4: Three Optimizations](#milestone-4-three-optimizations)
* [Final Submission](#final-submission)
* [Rubric](#rubric)
* [Final Report](#final-report)

## Milestone 1: Team Registration

| Deliverables |
| ------------ |
| Register your team in the google sheet. |

You and your team should agree on a team name and enter it in this [google sheet](https://forms.gle/7UU8X3wNvVGaV5ae6)

## Milestone 2: Rai Installation, CPU convolution

***Individual Submission***

As with all milestones, you will include an updated PDF `report.pdf` in the project directory you submit with rai.
This report should contain all of the deliverables.
This report should contain your names, netids, team names, and school affiliation (Illinois or ZJUI students). Please read the [Final Report](#final-report) section before submitting your report for the milestones.

| Deliverables |
| ------------ |
| Create a CPU convolution implementation |
| Report: Show output of rai running Mini-DNN on the CPU (CPU convolution implemented) for batch size of 10k images|
| Report: List Op Times (CPU convolution implemented) for batch size of 10k images|
| Report: List whole program execution time (CPU convolution implemented) for batch size of 10k images|
| Use `rai -p <project folder> --queue rai_amd64_ece408 --submit=m2` to mark your job for grading |

Clone this repository to get the project folder.

    git clone -b 2020fa https://github.com/illinois-impact/ece408_project.git

Download the rai binary for your platform from [here](https://drive.google.com/drive/folders/1Pp84x3So9OEHUwRHQVZcRP441wRsO-UV). 
You will probably use it for development, and definitely use it for submission. After downloading the rai binary, rename it to `rai` so that it is consistent with the instructions in this document. Also give `rai` execute permission by running in the folder you placed it.

    chmod +x rai

Note that you will have to run `rai` from wherever you placed it in your filesystem. For e.g., if you are running it from the same directory it is placed, run

    ./rai

You should have received a `.rai_profile` file by email.
Put that file in `~/.rai_profile` (Linux/macOS).
Your `.rai_profile` should look something like this (indented with space!)

    profile:
        firstname: <your-given-name>
        lastname: <your-surname>
        username: <your-username>
        email: <your-institution-email>
        access_key: <your-access-key>
        secret_key: <your-secret-key>
        affiliation: uiuc
        team:
            name: ""

*Note that it is okay for your team name to be blank. We track your team assignments using a database in the backend*

Some more info is available on the [Client Documentation Page](https://github.com/rai-project/rai).

### Testing Rai
Run the default Mini-DNN forward pass using rai without any CPU/GPU implementation.

Use RAI to run a batch forward pass on some test data.

    rai -p <project-folder> --queue rai_amd64_ece408

Note that the `<project-folder>` path should point to the root of this repository.

This will upload your project directory to rai and move it to `/src`, where the execution specified in `rai_build.yml` will occur. 

***Understanding rai_build.yml***

The `image:` key specifies the environment that the rest of the execution will occur in.
This environment includes the Mini-DNN framework as well as the model definition and pre-trained weights that will be used to do inference. **(Do not modify this entry)**

The `resources:` key specifies what computation resources will be available to the execution. **(Do not modify this entry)**

The `commands:` key specifies the recipe that rai will execute. First, the project files are copied to the `/build/student_code` directory so that we have a record of your code along with your performance.
Then the files in `custom` are copied to `/ece408/project/src/layer/custom` in the Mini-DNN source tree and the pretrained weights are copied to `/build`. Finally, Mini-DNN is recompiled with your custom code.

`./m2 100` runs the code specified in `m2.cc` program for a batch of 100 input images. 

You should see the following output:

    ✱ Running /bin/bash -c "./m2 100"
    Test batch size: 100
    Loading fashion-mnist data...Done
    Loading model...Done
    Conv-CPU==
    Op Time: 0.000655 ms
    Conv-CPU==
    Op Time: 0.000246 ms
    Test Accuracy: 0.08

It is okay for the accuracy is low here since you haven't implemented the convolutional layers yet.

Modify `rai_build.yml` to use `time` to measure the elapsed time of the whole program.

    - /bin/bash -c "time ./m2 100"

### Create a CPU Implementation

See the [description](#skeleton-code-description) of the skeleton code for a brief overview of what each file does.

Modify `custom/cpu-new-forward.cc` to implement the forward convolution described in Chapter 16 of the textbook.
The performance of the CPU convolution is not part of the project evaluation. We only evaluate for correctness.

The algorithm is also below, for your convenience

    for b = 0 .. B                     // for each image in the batch 
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

Unlike the convolutions described in the class, note that this one is not centered on the input image. There is no padding and the strides are 1. The following illustration may help you visualize this better.

![ConvExample](https://stanford.edu/~shervine/teaching/cs-230/illustrations/convolution-layer-a.png?1c517e00cb8d709baf32fc3d39ebae67)

*Source: https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks#layer*

Modify `rai_build.yml` to invoke

    - /bin/bash -c "./m2"

Please be patient as the CPU implementation with 10k images is slow. A correct implementation may take around 6-7 mins to run. If you want to iterate quickly when developing code using smaller batch sizes, see [Specifying Batch Size](#specifying-batch-size). When your implementation is correct, you should see output like this:

    Test batch size: 10000
    Loading fashion-mnist data...Done
    Loading model...Done
    Conv-CPU==
    Op Time: 74253.3 ms
    Conv-CPU==
    Op Time: 213714 ms
    Test Accuracy: 0.8714

Every time your layer is invoked, it will print the "Op Time," the time spent working on that layer.
Since the network has two convolutional layers, two times will be printed.
You can time the whole program execution by modifying `rai_build.yml` with

    - /bin/bash -c "time ./m2"

### Specifying Batch Size
`./m2`, `./m3`, `./m4` and `./final` all take one optional argument: the dataset size.  
If the correctness for each possible batch size is as below, you can be reasonably confident your implementation is right. The correctness does depend on the data size. 

For example, to check your accuracy on the full data size of 10,000, you could modify `rai_build.yml` to run

    - /bin/bash -c "./m2 10000"

| Number of Images | Accuracy  |
| -----------------| --------- |
| 100              | 0.86 |
| 1000             | 0.886 |
| 10000 (default)  | 0.8714 |

The provided `m2.cc` is identical to the one used by `--submit=m2`.

Use

    rai -p <project folder> --queue rai_amd64_ece408 --submit=m2

to mark your submission. Make sure to include your `report.pdf` in your `<project folder>`.

## Milestone 3: Baseline Convolutional Kernel

***Individual Submission***

| Deliverables |
| ------------ |
| Everything from Milestone 2 |
| Implement a GPU Convolution |
| Correctness and timing with 3 different dataset sizes |
| Report: Show output of rai running your GPU implementation of convolution (including the OpTimes) |
| Report: Demonstrate `nsys` profiling the GPU execution |
| Report: Include a list of all kernels that collectively consume more than 90% of the program time. |
| Report: Include a list of all CUDA API calls that collectively consume more than 90% of the program time. |
| Report: Include an explanation of the difference between kernels and API calls |
| Report: Screenshot of the GPU SOL utilization in Nsight-Compute GUI for your kernel profiling data|
| Use `rai -p <project folder> --queue rai_amd64_ece408 --submit=m3` to mark your job for grading |

### Create a GPU Implementation

Modify `custom/new-forward.cu` to create GPU implementation of the forward convolution.

Modify `rai_build.yml` to run

    - /bin/bash -c "./m3"

to use your GPU implementation.
When it is correct, it will show the same correctness as Milestone 2. To quicken development time, `m3.cc` takes one optional argument: the dataset size. See [Specifying Batch Size](#specifying-batch-size).

### Use Nsight-Systems and Nsight-Compute for initial Performance Results

First, ensure you are using correct image in rai_build.yml file

`image: illinoisimpact/ece408_minidnn_docker:amd64-gpu-cu10.2-fa20`

**Before you do any profiling, make sure you do not have any memory errors by running `cuda-memcheck`. See [Checking for Errors](#checking-for-errors) on how to run this.**

***System level profiling using Nsight-Systems***

We will learn how to use `nsys` (Nsight Systems) to profile the execution at the application level.

Once you've gotten the appropriate accuracy results, generate a profile using `nsys`. Make sure `rai_build.yml` is configured for a GPU run. Then, modify `rai_build.yml` to generate a profile instead of just executing the code.

    - nsys profile --stats=true ./m3

You should see something that looks like the following (but not identical):

~~~bash 
Collecting data...
Test batch size: 10000
Loading fashion-mnist data...Done
Loading model...Done
...
Generating CUDA API Statistics...
CUDA API Statistics (nanoseconds)

Time(%)  Total Time  Calls      Average   Minimum    Maximum  Name            
-------  ----------  -----  -----------  --------  ---------  ----------------
   72.3   294859478      2  147429739.0    675112  294184366  cudaMalloc      
   22.8    92865680      2   46432840.0  44841150   48024530  cudaMemcpy      
    4.5    18405301      2    9202650.5     25789   18379512  cudaLaunchKernel
    0.4     1467989      2     733994.5    473054     994935  cudaFree
Generating CUDA Kernel Statistics...

Generating CUDA Memory Operation Statistics...
CUDA Kernel Statistics (nanoseconds)

Time(%)  Total Time   Instances  Average  Minimum    Maximum  Name                
-------  ----------  ----------  -------  -------  ---------  --------------------
  100.0        3360           2   1680.0     1664       1696  conv_forward_kernel 


CUDA Memory Operation Statistics (nanoseconds)

Time(%)  Total Time  Operations     Average   Minimum   Maximum  Name              
-------  ----------  ----------  ----------  --------  --------  ------------------
  100.0    89602913           2  44801456.5  41565528  48037385  [CUDA memcpy HtoD]


CUDA Memory Operation Statistics (KiB)

   Total  Operations   Average     Minimum   Maximum  Name              
--------  ----------  --------  ----------  --------  ------------------
538906.0           2  269453.0  250000.000  288906.0  [CUDA memcpy HtoD]

~~~

The CUDA API Statistics section shows the CUDA API calls that are executed. The CUDA Kernel Statistics lists all the kernels that were executed during the profiling session. There are also more details on the CUDA memory operations (CudaMemcpy) listed.
There are columns corresponding to percentage of time consumed, total time, number of calls, and average/min/max time of those calls. Answer the questions stated in the deliverables using **your** `nsys` profiling result in the report as well as the `nsys` output corresponding to the section shown above.

Think about the distinction between a CUDA API call and a kernel launch, and describe it briefly in your report.
The CUDA documentation describes [kernels](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#kernels) and the [programming interface](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-interface).

You can find more information about `nsys` in the [Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/UserGuide/#cli-profiling)

***Kernel level profiling using Nsight-Compute***

Nsight-Systems does not give you detailed kernel level performance metrics. For that, we will need to use `nv-nsight-cu-cli` (Nsight-Compute). 

Modify `rai_build.yml` to use `nv-nsight-cu-cli` to save some timeline and analysis information, as described in [profiling](#profiling).
Use the NVIDIA Nsight Compute GUI to find the execution of your kernel, and show a screen shot of the GPU SOL utilization in your report.
The [Nsight Compute installation](#nsight-compute-installation) section describes how to install Nsight-Compute GUI on your personal machine. Note that you do not need CUDA to be installed. 

Use

    rai -p <project folder> --queue rai_amd64_ece408 --submit=m3

to mark your submission. Make sure to include your `report.pdf` in your `<project folder>`.

## Milestone 4: Three Optimizations

***Group Submission***

| Deliverables |
| ------------ |
| Everything from Milestone 3 |
| Implement three GPU optimizations |
| Report: Describe the optimizations as specified [here](#final-report) |
| Report: Use data from `nsys` and/or `nv-nsight-cu-cli` to analyze your optimizations and justify the effects of your optimizations |
| Use `rai -p <project folder> --queue rai_amd64_ece408 --submit=m4` to mark your job for grading |

### Interpreting the timing output from rai

You will see three types of times reported per layer as follows

    ✱ Running bin/bash -c "./m4 1000"   \\ Output will appear after run is complete.
    Test batch size: 1000
    Loading fashion-mnist data...Done
    Loading model...Done
    Conv-GPU==
    Conv-GPU==

    Test Accuracy: 0.886
    --------------------------------
    -           TIMINGS
    --------------------------------
    Layer 1 GPUTime: 4.862905 ms
    Layer 1 OpTime: 4.882009 ms
    Layer 1 LayerTime: 63.512461 ms
    Layer 2 GPUTime: 16.523901 ms
    Layer 2 OpTime: 16.544541 ms
    Layer 2 LayerTime: 59.37215 ms


1. GPUTime - This is the total time your kernel code takes to execute on the GPU. It does not include any CUDA API calls
2. OpTime - This is time between the last cudaMemcpy call before your first kernel call and the first cudaMemcpy after your last kernel call in `new-forward.cu -> conv_forward_gpu()`. It does not include the cudaMemcpy times. **This is the time that will be used for rankings**
3. LayerTime - This is the total time taken to execute `new-forward.cu -> conv_forward_gpu()`. It includes all the times for all kernel and CUDA API calls as well as the CPU time to run `conv_forward_gpu()`.

*Note: We will check that GPUTime < OpTime < LayerTime and that the GPUTime and OpTime should be simillar. Also LayerTime should not differ from OpTime by more than 1000 ms.*

*Only OpTImes of runs with batch size set as 10k images will be considered for ranking.*

### 4.1 Add three GPU Optimization

For this milestone, you should attempt at least three GPU optimizations (see [optimizations](#optimizations)).

Describe the optimizations in your `report.pdf`. Read the [Final Report](#final-report) section to see what is expected for the description of each optimization.

### 4.2 Performance Analysis with Nsight-Systems and Nsight-Compute

Use the NVIDIA Nsight-Systems(`nsys`) and Nsight-Compute(`nv-nsight-cu-cli`) and your analysis information to describe the effect that your optimizations had on the performance of your convolution.
If possible, you should try to separate the effect of each optimization in your analysis.

Use 
    
    rai -p <project folder> --queue rai_amd64_ece408 --submit=m4
    
to submit your project folder. Make sure to include your `report.pdf` in your `<project folder>`.

## Final Submission

***Group Submission***

| Deliverables |
| ------------ |
| Everything from Milestone 4 |
| Implement final GPU optimizations  (total of 6) |
| Report: Describe and analyze the optimizations |
| Report: Use `nsys` and/or `nv-nsight-cu-cli` to justify the effects of your optimiatization on performance |
| Use `rai -p <project folder> --queue rai_amd64_ece408 --submit=final` to mark your job for grading |

### Optimized Layer

Optimize your GPU convolution (see [optimizations](#optimizations)).

Your implementation must work with `rai -p <project-folder> --queue rai_amd64_ece408 --submit=final`.
This means all your source files must be in `custom/`, and your implementation must work when they are copied to `/ece408/project/src/layer/custom` in the Mini-DNN tree, and `make` is invoked on the Mini-DNN tree. This is done in the provided `rai_build.yml`.

Use `rai -p <project folder> --queue rai_amd64_ece408 --submit=final` to submit your project folder. Make sure to include your `report.pdf` in your `<project folder>`.

### Final Report

You've been building this final report through all the milestones.
Keep the content from the earlier milestones, but be sure to include the following:

* Your team member names
* Your netids
* Your team name
* Your school affiliation (Illinois or ZJUI students)

The final report should include at least the following information for each optimization

1. **Optimization Approach and Results**
    * how you identified the optimization opportunity
    * why you thought the approach would be fruitful
    * the effect of the optimization. was it fruitful, and why or why not. Use `nsys` and `nv-nsight-cu` to justify your explanation.
    * Any external references used during identification or development of the optimization
    * How  your team organized and divided up this work.
2. **References** (as needed)
3. **(Optional) Suggestions for Improving Next Year**

### Rubric

The overall project score will be computed as follows:

1. Milestone 1 ( 5% )
2. Milestone 2 ( 10% )
3. Milestone 3 ( 10% )
4. Milestone 4 ( 30% )
    * Optimization 1 ( 10% )
    * Optimization 2 ( 10% )
    * Optimization 3 ( 10% )
5. Final Optimizations ( 30% )
    * Optimization 4 ( 10% )
    * Optimization 5 ( 10% )
    * Optimization 6 ( 10% )
    * Additional Optimizations / detailed insights ( up to +10% extra!!! )
6. Performance Ranking ( 10% )
7. Report Style (5 %)
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
The ranking is determined solely by the values printed by `Op Time:` during your run. See [here](#Interpreting-the-timing-output-from-rai) for details.

## Optimizations

We are going to suggest a set of possible optimizations for you to attempt.

* Unroll + shared-memory Matrix multiply
* Shared Memory convolution
* Kernel fusion for unrolling and matrix-multiplication
* Weight matrix (kernel values) in constant memory
* Tuning with restrict and loop unrolling (considered as one optimization only if you do both)
* An advanced matrix multiplication algorithm (register-tiled, for example)
* Sweeping various parameters to find best values (block sizes, amount of thread coarsening)
* Exploiting parallelism in input images, input channels, and output channels.
* Multiple kernel implementations for different layer sizes
* Input channel reduction: tree
* Input channel reduction: atomics
* Fixed point (FP16) arithmetic
* Using Tensor Cores to speed up matrix multiplication
* ...

Other optimizations that do not fit in here may also be considered as optimizations.
If in doubt, contact the course staff.

### Checking for Errors

Within `custom/new-forward.cu`, you can use the predefined error handling code to catch CUDA errors or, you can define a macro/function similar to `wbCheck` used in WebGPU.

To catch memory errors, prepend your command with `cuda-memcheck`

    - /bin/bash -c "cuda-memcheck ./m3"

### Profiling

You can gather system level performance information using `nsys`.
For detailed kernel level GPU profiling, use `nv-nsight-cu-cli` and view that information with `nv-nsight-cu`.

You can see some simple information like so (as we did in milestone 3):

    nsys profile --stats=true <your command here>

You can additionally gather some detailed kernel level performance metrics.

    nv-nsight-cu-cli --section '.*' -o analysis_file <your command here>

This will generate `analysis_file.ncu-rep`.
`--section '.*'` may significantly slow the run time since it is profiling all the metrics. You may wish to modify the command to run on smaller datasets during this profiling.

You will need to follow the link rai prints after the execution to retrieve these files.
You can use the NVIDIA Nsight Compute GUI (`nv-nsight-cu`) to import those files.
You will need to install NVIDIA NSight Compute on your own machine. It can be downloaded as a standalone application. See instructions [here](#nsight-compute-installation)

To import the files:
* Launch the GUI `/usr/local/NVIDIA-Nsight-Compute/nv-nsight-cu` (or from wherever you installed it)
* Close the intial Quick Launch menu
* Go to File > Open File and select the `.ncu-rep` file from the `\build` folder you downloaded from rai.

*OR*
* Directly launch from the terminal `/usr/local/NVIDIA-Nsight-Compute/nv-nsight-cu <filename>.ncu-rep`

For a high-level overview of the Nsight software, visit [here](https://developer.nvidia.com/tools-overview).

### Nsight-compute Installation

Nsight-Compute can be installed as a standalone application. You do not need CUDA to be installed. You can download the installer from NVIDIA's [website](https://developer.nvidia.com/gameworksdownload#?dn=nsight-compute-2020-2-0)

### Skeleton Code Description
`custom/cpu-new-forward.cc` and `custom/new-forward.cu` containes skeleton implementations for the CPU and GPU convolutions respectively. You can complete the project by modifying these two files only. `custom/cpu-new-forward.h` and `custom/gpu-new-forward.h` are the respective header files. You need not modify these files unless you need to declare your own functions.

The code in `m2.cc`, `m3.cc`, `m4.cc` and `final.cc` are the top level files that are executed for each milestone. You should not be modifying these files.

## License

NCSA/UIUC © 2020 [Carl Pearson](https://cwpearson.github.io)

## Contributors

* [Carl Pearson](https://cwpearson.github.io)
* [Vikram Mailthody](https://github.com/msharmavikram/)
* Andrew Schuh
* Abdul Dakkak
* Zaid Qureshi
* Rui Lan
* Zhicun Wan
* Ben Schreiber
* James Cyriac

