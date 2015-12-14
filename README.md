#### Harvard CS205 Final Project - Parallel Point Based Value Iteration Algorithm for Solving POMDPs
============================================

### Authors

Jaemin Cheun (jcheun@college.harvard.edu)

### Project Website

http://cs205jcheun.weebly.com/

### Video

https://www.youtube.com/watch?v=_abCt_ew13o


### Background and Motivation

POMDP is a model for formalizing decision problems under uncertainty. POMDP uses “belief-based” programming, expressing what we know about the uncertain aspects of the world as a “belief state,”  a probability distribution over possible world states, and then selects actions based on the current belief state. 

In this project, our goal is to develop an efficient algorithm for PBVI algorithm. Point Based Value Iteration (PBVI) gains computational advantage by applying value updates at few and reachable belief points rather than over the entire belief space. 
When updating hyperplanes in the PBVI algorithm, all belief points are independent of the other points in B, thus the PBVI algorithm is well suited for parallelization. When implementing the GPU program, we need to choose the parameter to parallelize. Because the second part of the algorithm (the backup part) and because each belief point is independent, we parallelize over B. In the project, I implemented the algorithm both using the global memory and the shared memory and compared the two.

### Description

This project explores different parallel implementations of PBVI algorithm and compares the performance of them with the serial version in c++. Number of Blocks can be specified.


### Shortcomings

I spent a lot of time trying to work with many different platforms such as PyCUDA, PyOpenCL, and NumbaPRO to leverage my knowledge in Python, but they did not work on my computer somehow. So eventhough I'm not as comfortable with C++, to leaverage the CUDA C++/C programming language, I decided to try out C++. It took a while to figure out how the language works, and eventhough I tried to sperate the files as much as I can, I somehow got an error everytime. Therefore, I decided to put all the code in one file (I'm really sorry), just to make sure that everything works as planned.


### Code Instructions


Please download the appropriate CUDA Toolkit packet from https://developer.nvidia.com/cuda-toolkit.
Then type in the directory: 

export LD_LIBRARY_PATH=/usr/local/cuda/lib

export PATH=$PATH:/usr/local/cuda/bin

and then run:

nvcc main.cu -o main.out

./main.out

You can change the number of belief counts, number of iteratinos, and number of blocks by changing the code in main.cu. 
Just change the number next to #define NUM_BELIEF etc.

### Problem Space

We tested the algorithm on a simple POMDP problem domain called the Tiger Problem. 

Imagine an agent standing in front of two closed doors. Behind one of the doors is a tiger and behind the other is a large reward. If the agent opens the door with the tiger, then a large penalty is received (presumably in the form of some amount of bodily injury). Instead of opening one of the two doors, the agent can listen, in order to gain some information about the location of the tiger. Unfortunately, listening is not free; in addition, it is also not entirely accurate. There is a chance that the agent will hear a tiger behind the left-hand door when the tiger is really behind the right-hand door, and vice versa.




### Machine Used

The platforms detected are:

Apple Apple version: OpenCL 1.2 (Sep 21 2015 19:24:11)
The devices detected on platform Apple are:

Intel(R) Core(TM) i7-3615QM CPU @ 2.30GHz [Type: CPU ]
Maximum clock Frequency: 2300 MHz
Maximum allocable memory size: 2147 MB
Maximum work group size 1024
Maximum work item dimensions 3
Maximum work item size [1024L, 1L, 1L]

HD Graphics 4000 [Type: GPU ]
Maximum clock Frequency: 1200 MHz
Maximum allocable memory size: 402 MB
Maximum work group size 512
Maximum work item dimensions 3
Maximum work item size [512L, 512L, 512L]

GeForce GT 650M [Type: GPU ]
Maximum clock Frequency: 900 MHz
Maximum allocable memory size: 268 MB
Maximum work group size 1024
Maximum work item dimensions 3
Maximum work item size [1024L, 1024L, 64L]

MacBook Pro (Retina, Mid 2012)
Processor 2.3 GHz Intel Core i7
Memory 8 GB 1600 MHz DDR3
Graphics Intel HD Graphics 4000 1536 MB

### Result

Please check the website!


### Acknowledge

We thank Ray and all CS205 TFs for providing the wonderful course and all helpful instructions.
