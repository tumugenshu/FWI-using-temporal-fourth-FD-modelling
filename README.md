## FWI-using-temporal-fourth-FD-modelling

These codes are used to complete elastic full waveform inversion using 
temporal fourth-order FD modelling. In order to accelete the algorith, 
GPU-based versions are privided here. What's more, the GPU shared memory
 is used in block level and the efficiency improvement is about 10%~15% 
on GTX 750ti, where efficiency improvement is dependent on the length of
 FD orders.

#The codes contains two parts: one is the GPU-based elastic FWI using 
temporal fourth-order Finite-different modelling method, the other is 
that we use GPU-shared-memory to optimize the first one further. The 
tested GPU device is GTX 750ti. The FWI section are contains forward 
section, which can be used to output the forward seismograms.

# How to run the code 

Firstly, install the MPICH and Cuda, and set the path correctly on your 
~/.bashrc file and source it. Then modify the mpicc and nvcc paths in 
the Makefile in the package. Then compile and generate executable file, 
run it by sh run.sh.

# main files in the package
Seven files are:
1. *.cpp,main control code for elastic FWI by MPI. It includes the 
functions of domain decomposition, multi-scale and encoding number 
parameters, and the MPI data exchange, model parameter update, and etc.
2. *.cu, the CUDA code and designed for simultaneous source propagation.
 The main four GPU kernel functions are forward, reconstructed, backward
 sections.
3. headmulti.h, used for declaring all the functions to be called in the
 elastic_fdtd_3d_FWI_ModelCMultiscale.cpp.
4. Makefile, used for compiling the program with the CUDA’s nvcc, in 
addition, some shared library functions are listed in the file. One 
thing to be mentioned is that we should make sure the paths of cuda and 
mpi are set correctly in this file.
5. hostgpu, a name list of nodes of GPUs.
6. run.sh, used to run the elastic FWI on the nodes listed in hostsgpu 
by mpiexec or mpirun.
7. nohup.out, a log file used to write the running status. Created 
automatically by running
run.sh.
Two folders are:
1. input/, used to save initial P- and S-wave velocity and density which
 are binary and the stored along the depth direction. 
2. output/, is used to store the inversion information, including the 
inversions of different frequency bands, objective function misfit, and 
temporal results, such as the synthetic data, residual data 
andgradients.