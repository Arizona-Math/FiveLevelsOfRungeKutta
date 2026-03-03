# Runge-Kutta Numerical Integration Project

## Overview
This project implements various algorithms for solving ordinary differential equations (ODEs) using different approaches: CUDA, OpenMP, MPI, and sequential C++. Each implementation showcases the versatility and performance of the Runge-Kutta method in different computational contexts.

## Files and Summaries

### 1. Makefile
Defines compilation rules for C++ and CUDA implementations.

### 2. rk4.cu
Implements a CUDA kernel to solve an ODE using the Runge-Kutta method on the GPU.

### 3. rk4.py
Python implementation that mimics the functionality of the C++ CUDA code.

### 4. rk4_mpi.cpp
MPI implementation for parallel execution across multiple processes.

### 5. rk4_no_cuda.cpp
Sequential C++ implementation of the Runge-Kutta method without GPU acceleration.

### 6. rk4_no_cuda_omp.cpp
C++ implementation that uses OpenMP for parallel processing.

## Overall Summary
The project successfully implements various Runge-Kutta algorithms.

## Contribution
Contributions are welcome! Please feel free to submit pull requests or issues for further improvements.

## License
This project is licensed under the MIT License.

