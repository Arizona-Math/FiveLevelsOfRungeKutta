#include <cuda_runtime.h>
#include <cuComplex.h>
#include <stdint.h>
#include <stdio.h>

__device__ __forceinline__
double rk4_step(double t, double y)
{
  // Zero vectorfield
  return y;
}

__global__
void apply_rk4(double* state, double* t, uint32_t dim)
{
  const double h=1e-4;
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= dim) return;
  state[i] = rk4_step(*t, state[i]);
  *t += h;
}

/* ------------------ tiny demo harness ------------------ */

static void check_cuda(cudaError_t e, const char* msg)
{
  if (e != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s: %s\n", msg, cudaGetErrorString(e));
    exit(1);
  }
}

int main()
{
  const uint32_t N = 10000;		// A million

  // Allocate statevector on device
  double* d_state = nullptr;
  double* t = nullptr;
  check_cuda(cudaMalloc(&d_state, N * sizeof(double)), "cudaMalloc state");
  check_cuda(cudaMalloc(&t,       N * sizeof(double)), "cudaMalloc t");  

  // Initialize statevector to something simple on device
  check_cuda(cudaMemset(d_state, 1, dim * sizeof(double)), "cudaMemset");

  // Apply kernel
  const int threads = 256;
  const int blocks = (dim + threads - 1) / threads;
  apply_rk4<<<blocks, threads>>>(d_state, t, dim);
  check_cuda(cudaGetLastError(), "kernel launch");
  check_cuda(cudaDeviceSynchronize(), "kernel sync");

  // Copy back a few entries to sanity-check
  double h0, h1, h2, h3;
  check_cuda(cudaMemcpy(&h0, d_state + 0, sizeof(double), cudaMemcpyDeviceToHost), "copy 0");
  check_cuda(cudaMemcpy(&h1, d_state + 1, sizeof(double), cudaMemcpyDeviceToHost), "copy 1");
  check_cuda(cudaMemcpy(&h2, d_state + 2, sizeof(double), cudaMemcpyDeviceToHost), "copy 2");
  check_cuda(cudaMemcpy(&h3, d_state + 3, sizeof(double), cudaMemcpyDeviceToHost), "copy 3");

  printf("amp[0]=%g\n", h0);
  printf("amp[1]=%g\n", h1);
  printf("amp[2]=%g\n", h2);
  printf("amp[3]=%g\n", h3);

  cudaFree(d_state);
  return 0;
}
