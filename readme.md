# ECE408/CS483 Final Project

## Introduction

This is the skeleton code for the Fall 2021 ECE408 / CS483 / CSE408 course project.

In this final project, you will be implementing and optimizing the forward-pass of a convolutional layer using CUDA. Convolutional layers are the primary building blocks of convolutional neural networks (CNNs), which are used in many machine learning tasks like image classification, object detection, natural language processing, and recommendation systems. In general, CNNs work well on tasks where the data/input features have some level of spatial relationship.

You will be working with a **modified** version of the LeNet-5 architecture shown below.

![LenetImage](https://lh5.googleusercontent.com/84RlneM7JSDYDirUr_ceplL4G3-Peyq5dkLJTe2f-3Bj9KuWZjsH2A9Qq5PO5BRLrVfWGPnI3eQu8RkTPgyeUf9ZOWY9JbptVJy9LceAyHRn-O0kbzprx88yb82a5dnCR7EDP7n0)

*Source: http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf*

Your optimized CUDA implementation of the convolutional layer will be used to perform inference for layers C1 and C3 (shown in red) in the figure above. We will be leveraging the [mini-dnn-cpp](https://github.com/iamhankai/mini-dnn-cpp) (Mini-DNN) framework for implementing the modified LeNet-5. 

We will be using the [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist), where the inputs to the network will be a batch of 10,000 single channel images, each with dimensions of 86 x 86 pixels. The output layer consists of 10 nodes, where each node represents the likelihood of the input belonging to one of the 10 classes (T-shirt, dress, sneaker, boot etc.)

The overall learning objectives for this project are:
* Demonstrating command of CUDA and optimization approaches by designing and implementing an optimized neural-network convolutional layer forward pass
* Obtaining practical experience in analyzing and fine tuning CUDA kernels through the use of profiling tools like Nsight Systems (`nsys`) and Nsight-Compute (`nv-nsight-cu`)

You will be working on this project individually.

*You are expected to adhere to University of Illinois academic integrity standards. Do not attempt to subvert any of the performance-measurement aspects of the final project. If you are unsure about whether something does not meet those guidelines, ask a member of the teaching staff.*

## Table of Contents

* [Milestone 1: Rai Installation, CPU Convolution, Profiling](#milestone-1-rai-installation-cpu-convolution-profiling)
* [Milestone 2: Baseline Convolutional Kernel](#milestone-2-baseline-convolutional-kernel)
* [Milestone 3: GPU Convolution Kernel Optimizations](#milestone-3-gpu-convolution-kernel-optimizations)
* [Rubric](#rubric)
* [Optimizations](#optimizations)
* [Appendix](#appendix)

## Milestone 1: Rai Installation, CPU convolution, Profiling

***Deadline: October 15th, 8 PM CST***

For each milestone, you will include a PDF `report.pdf` file in the project directory you submit with rai. You can create this report by filling out the template report (.docx) that we have provided for each milestone and exporting it as a PDF file.


| Deliverables |
| ------------ |
| Create a CPU convolution implementation |
| Profile your implementation with `gprof` |
| Write your report |
| Use `./rai -p <project folder> --queue rai_amd64_ece408 --submit=m1` to mark your job for grading |

Clone this repository to get the project folder.

    git clone -b 2021fa https://github.com/illinois-impact/ece408_project.git

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
        username: <your-netid>
        email: <your-institution-email>
        access_key: <your-access-key>
        secret_key: <your-secret-key>
        affiliation: uiuc
        role: ece408
            team: <your-netid>

Some more info is available on the [Client Documentation Page](https://github.com/rai-project/rai).

### Testing Rai
Run the default Mini-DNN forward pass using rai without any CPU/GPU implementation.

Use RAI to run a batch forward pass on some test data.

    ./rai -p <project-folder> --queue rai_amd64_ece408

Note that the `<project-folder>` path should point to the root of this repository.

This will upload your project directory to rai and move it to `/src`, where the execution specified in `rai_build.yml` will occur. 

***Understanding rai_build.yml***

The `image:` key specifies the environment that the rest of the execution will occur in.
This environment includes the Mini-DNN framework as well as the model definition and pre-trained weights that will be used to do inference. **(Do not modify this entry)**

The `resources:` key specifies what computation resources will be available to the execution. **(Do not modify this entry)**

The `commands:` key specifies the recipe that rai will execute. First, the project files are copied to the `/build/student_code` directory so that we have a record of your code along with your performance.
Then the files in `custom` are copied to `/ece408/project/src/layer/custom` in the Mini-DNN source tree and the pretrained weights are copied to `/build`. Finally, Mini-DNN is recompiled with your custom code.

`./m1 100` runs the code specified in `m1.cc` program for a batch of 100 input images. 

You should see the following output:

    ✱ Running /bin/bash -c "./m1 100"
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

    - /bin/bash -c "time ./m1 100"

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

    - /bin/bash -c "./m1"

Please be patient as the CPU implementation is slow and will take several minutes to run. (For instance, a correct implementation with 10k images may take 13+ mins to run). If you want to iterate quickly when developing code using smaller batch sizes, see [Specifying Batch Size](#specifying-batch-size). When your implementation is correct, you should see output like this:

    Test batch size: 1000
    Loading fashion-mnist data...Done
    Loading model...Done
    Conv-CPU==
    Op Time: 7425.3 ms
    Conv-CPU==
    Op Time: 21371.4 ms
    Test Accuracy: 0.886

Every time your layer is invoked, it will print the "Op Time," the time spent working on that layer.
Since the network has two convolutional layers, two times will be printed. Note, these times will vary slightly between every run due to several factors, such as RAI server use.
You can time the whole program execution by modifying `rai_build.yml` with

    - /bin/bash -c "time ./m1"

### Specifying Batch Size
`./m1`, `./m2`, and `./m3` all take one optional argument: the dataset size.  
If the correctness for each possible batch size is as below, you can be reasonably confident your implementation is right. The correctness does depend on the data size. 

For example, to check your accuracy on the full data size of 10,000, you could modify `rai_build.yml` to run

    - /bin/bash -c "./m1 10000"

| Number of Images | Accuracy  |
| -----------------| --------- |
| 100              | 0.86 |
| 1000             | 0.886 |
| 10000            | 0.8714 |

Note: Due to the limited capacity of our RAI servers, in order to ensure RAI job submissions take a reasonable amount of time, we are only requiring you to run and profile your CPU implementation with a batch size of 1000 images for this milestone. Please do not run milestone one with a batch size of 10000.

### Use Gprof to profile your CPU implementation

You will use `gprof` to profile the execution of your CPU forward convolution implementation.

We compile and link your `cpu-new-forward.cc` with the `-pg` flag, which creates a `gmon.out` artifact containing profile information when the binary `m1` is executed.  To analyze this information in human readable form, modify `rai_build.yml` and add the line
 
    - /bin/bash -c "gprof -Q m1 gmon.out"

By default, `gprof` prints both a flat profile and a call graph (see "Interpreting gprof's Output" in the [GNU gprof Documentation](https://sourceware.org/binutils/docs/gprof/index.html)).  With the `-Q` flag, we only print the flat profile.  The information you need can be found near the beginning of `gprof`'s output, so you can pipe the output to `grep` (with your function's name) or `head`.

For this milestone, edit the responses in the given `m1_report_template.docx` file, export the report as a PDF, and name the PDF as `report.pdf`.

| Report  |
| ------------ |
| Show output of rai running Mini-DNN on the CPU (CPU convolution implemented) for batch size of 1k images|
| List Op Times (CPU convolution implemented), whole program execution time, and accuracy for batch size of 1k images|
| Show percentage of total execution time of your program spent in your forward pass function with `gprof`|

Use

    ./rai -p <project folder> --queue rai_amd64_ece408 --submit=m1

to mark your submission for grading. Make sure to include your `report.pdf` in your `<project folder>`. Make sure you answer all items listed above for this milestone, and include your name, NetID, and class section.

## Milestone 2: Baseline Convolutional Kernel

***Deadline: November 5th, 8 PM CST***

| Deliverables |
| ------------ |
| Implement a GPU Convolution kernel |
| Verify correctness and record timing with 3 different dataset sizes |
| Write your report |
| Use `./rai -p <project folder> --queue rai_amd64_ece408 --submit=m2` to mark your job for grading |

### Create a GPU Implementation

First, make sure you have the most up-to-date version of the project repository by going to the project root directory and running:

    git fetch origin 2021fa
    git merge origin/2021fa

Modify `custom/new-forward.cu` to create GPU implementation of the forward convolution. This should be a basic convolution implement that does not include shared-memory or tiling.

Modify `rai_build.yml` to run

    - /bin/bash -c "./m2"

to use your GPU implementation.
When it is correct, it will show the same correctness as Milestone 1. To quicken development time, `m2.cc` takes one optional argument: the dataset size. See [Specifying Batch Size](#specifying-batch-size).

### Use Nsight-Systems and Nsight-Compute for initial Performance Results

First, ensure you are using correct image in rai_build.yml file

`image: jnativ/ece408_minidnn_docker_sp21:latest`

**Before you do any profiling, make sure you do not have any memory errors by running `cuda-memcheck`. See [Checking for Errors](#checking-for-errors) on how to run this.**

***System level profiling using Nsight-Systems***

We will learn how to use `nsys` (Nsight Systems) to profile the execution at the application level.

Once you've gotten the appropriate accuracy results, generate a profile using `nsys`. Make sure `rai_build.yml` is configured for a GPU run. Then, modify `rai_build.yml` to generate a profile instead of just executing the code.

    - nsys profile --stats=true ./m2

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
There are columns corresponding to percentage of time consumed, total time, number of calls, and average/min/max time of those calls. Use **your** `nsys` profiling output corresponding to the section above to answer the questions for your report.

Think about the distinction between a CUDA API call and a kernel launch, and describe it briefly in your report.
The CUDA documentation describes [kernels](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#kernels) and the [programming interface](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-interface).

You can find more information about `nsys` in the [Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/UserGuide/#cli-profiling)

***Kernel level profiling using Nsight-Compute***

Nsight-Systems does not give you detailed kernel level performance metrics. For that, we will need to use `nv-nsight-cu-cli` (Nsight-Compute). 

Modify `rai_build.yml` to use `nv-nsight-cu-cli` to save some timeline and analysis information, as described in [profiling](#profiling).
Use the NVIDIA Nsight Compute GUI to find the execution of your kernel, and show a screen shot of the GPU SOL utilization in your report.  You will see performance metrics for two kernel launches, one for each layer.
The [Nsight Compute installation](#nsight-compute-installation) section describes how to install Nsight-Compute GUI on your personal machine. Note that you do not need CUDA to be installed. 

For this milestone, edit the responses in the given `m2_report_template.docx` file, export the report as a PDF, and name the PDF as `report.pdf`.

| Report  |
| ------------ |
| Show output of rai running your GPU implementation of convolution (including the OpTimes) |
| Demonstrate `nsys` profiling the GPU execution |
| Include a list of all kernels that collectively consume more than 90% of the kernel time |
| Include a list of all CUDA API calls that collectively consume more than 90% of the API time |
| Include an explanation of the difference between kernels and API calls |
| Screenshot of the GPU SOL utilization in Nsight-Compute GUI for your kernel profiling data (for both kernel launches) |

Use

    ./rai -p <project folder> --queue rai_amd64_ece408 --submit=m2

to mark your submission for grading. Make sure to include your `report.pdf` in your `<project folder>`. Make sure you answer all items listed above for this milestone, and include your name, NetID, and class section.


## Milestone 3: GPU Convolution Kernel Optimizations

***Deadline: December 3rd, 8 PM CST***

| Deliverables |
| ------------ |
| Implement multiple GPU optimizations |
| Write your report |
| Use `./rai -p <project folder> --queue rai_amd64_ece408 --submit=m3` to mark your job for grading |

### Add GPU Optimizations

First, make sure you have the most up-to-date version of the project repository by going to the project root directory and running:

    git fetch origin 2021fa
    git merge origin/2021fa

You should attempt to implement at least 10 points of GPU optimizations (as seen in [optimizations](#optimizations)). You can implement these optimizations separately from each other or stack each optimization in order to maximize performance. If you implement your optimization separately, you must still include the code for each optimization in your submission even if it is unused in the final result. In this case it is recommended to create different methods and kernels to clarify what sections of the code apply to each optimization. 

You must also make sure to clarify which baseline is used when analyzing the performance for a new optimization. If you are analyzing a result with a single optimization implemented, you should compare against your basic convolution kernel in Milestone 2. If you begin to stack multiple optimizations, for each optimization you add should be compared against the previous version without said optimization. This way you can most efficently analyse the effects of adding the given optimization.

### Interpreting the timing output from rai

You will see two types of times reported per layer as follows


    ✱ Running bash -c "./m3 1000"   \\ Output will appear after run is complete.
    Test batch size: 1000
    Loading fashion-mnist data...Done
    Loading model...Done
    Conv-GPU==
    Layer Time: 61.1231 ms
    Op Time: 4.82135 ms
    Conv-GPU==
    Layer Time: 55.4437 ms
    Op Time: 16.6154 ms
    
    Test Accuracy: 0.886


1. "Op Time" - This is time between the last cudaMemcpy call before your first kernel call and the first cudaMemcpy after your last kernel call (i.e. just `new-forward.cu -> conv_forward_gpu()`). It does not include the cudaMemcpy times.
2. "Layer Time" - This is the total time taken to perform the convolution layer (C1 or C3). It includes the times for all kernel and CUDA API calls (i.e. the total time of all three `new-forward.cu -> conv_forward_gpu*` functions).

### Performance Analysis with Nsight-Systems and Nsight-Compute

Use the NVIDIA Nsight-Systems(`nsys`) and Nsight-Compute(`nv-nsight-cu-cli`) and your analysis information to describe the effect that your optimizations had on the performance of your convolution.
If possible, you should try to separate the effect of each optimization in your analysis.

For this milestone, edit the responses in the given `m3_report_template.docx` file, export the report as a PDF, and name the PDF as `report.pdf`. Describe in detail each optimization you implement, including how and why you choose to implement that specific optimization, why you thought the optimization may be fruitful, the actual results of the optimization and whether it was fruitful (use quantitative data from `nsys` and `nv-nsight-cu` to justify your explanation), and include any external references used during identification or development of the optimization.

| Report |
| ------------ |
| Describe the optimizations as specified |
| Use data from `nsys` and/or `nv-nsight-cu-cli` to analyze your optimizations and justify the effects of your optimizations |

Use 
    
    rai -p <project folder> --queue rai_amd64_ece408 --submit=m3
    
to submit your project folder. Make sure to include your `report.pdf` in your `<project folder>`. Make sure you answer all items listed above for this milestone, and include your name, NetID, and class section.

## Rubric

The overall project score will be computed as follows:

1. Milestone 1 ( 20% )
    * Correctness ( 15% )
    * Report ( 5% )
2. Milestone 2 ( 20% )
    * Correctness ( 15% )
    * Report( 5% )
3. Milestone 3 ( 60% )
    * Correctness ( 4% for each optimization point )
    * Report ( 2% for each optimization point )
4. Extra Credit ( up to +5% maximum )
    * Top 10 on leaderboard ( +5% )
    * Top 30 on leaderboard ( +3% )
    * Top 50 on leaderboard ( +1% )

This semester, ranking will be made available, via the `rai ranking` command.

## Optimizations

These are the list of optimizations we will consider valid for Milestone 3. You should implement 10 points worth of optimizations in order to recieve full credit for Milestone 3. If you would like to impelement a potential optimization that is not on this list, please consult a TA or instructor beforehand to verify that the optimization is valid and to assign it a point value.

* Tiled shared memory convolution (**2 points**)
* Shared memory matrix multiplication and input matrix unrolling (**3 points**)
* Kernel fusion for unrolling and matrix-multiplication (requires previous optimization) (**2 points**)
* Weight matrix (kernel values) in constant memory (**1 point**)
* Tuning with restrict and loop unrolling (considered as one optimization only if you do both) (**3 points**)
* Sweeping various parameters to find best values (block sizes, amount of thread coarsening) (**1 point**)
* Multiple kernel implementations for different layer sizes (**1 point**)
* Input channel reduction: tree (**3 point**)
* Input channel reduction: atomics (**2 point**)
* Fixed point (FP16) arithmetic. (note this can modify model accuracy slightly) (**4 point**)
* Using Streams to overlap computation with data transfer (**4 point**)
* An advanced matrix multiplication algorithm (register-tiled, for example) (**5 points**)
* Using Tensor Cores to speed up matrix multiplication (**5 points**)
* Overlap-Add method for FFT-based convolution (note this is **very** hard, and may not yeild a large performace increase due to mask size) (**8 points**)

## Appendix

### Checking for Errors

Within `custom/new-forward.cu`, you can use the predefined error handling code to catch CUDA errors or, you can define a macro/function similar to `wbCheck` used in WebGPU.

To catch memory errors, prepend your command with `cuda-memcheck`

    - /bin/bash -c "cuda-memcheck ./m3"

### Profiling

You can gather system level performance information using `nsys`.

For detailed kernel level GPU profiling, use `nv-nsight-cu-cli` and view that information with `nv-nsight-cu`.

You can see some simple information like so (as we did in milestone 2):

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
* Go to File > Open File and select the `.ncu-rep` file from the `\build` folder you downloaded from rai (note that the downloaded file is a `TAR` file, not a `TAR.GZ` as the name implies).

*OR*
* Directly launch from the terminal `/usr/local/NVIDIA-Nsight-Compute/nv-nsight-cu <filename>.ncu-rep`

For a high-level overview of the Nsight software, visit [here](https://developer.nvidia.com/tools-overview).

### Nsight-compute Installation

Nsight-Compute can be installed as a standalone application. You do not need CUDA to be installed. You can download the installer from NVIDIA's [website](https://developer.nvidia.com/gameworksdownload#?dn=nsight-compute-2020-3-0)

### Skeleton Code Description
`custom/cpu-new-forward.cc` and `custom/new-forward.cu` containes skeleton implementations for the CPU and GPU convolutions respectively. You can complete the project by modifying these two files only. `custom/cpu-new-forward.h` and `custom/gpu-new-forward.h` are the respective header files. You need not modify these files unless you need to declare your own functions.

The code in `m1.cc`, `m2.cc`, and `m3.cc` are the top level files that are executed for each milestone. You should not be modifying these files.

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
* Jonathan Nativ
* Henry Haase

