#include <cuda_runtime.h>
#include <cuComplex.h>
#include <stdint.h>
#include <stdio.h>


typedef std::pair<double, double> t_y;

__device__ __forceinline__
double rk4_step(double t, double y)
{
  // Zero vectorfield
  return y;
}

__global__
void apply_rk4(t_y* state, uint32_t dim)
{
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const double h=.1;

  if (i >= dim) return;


  state[i].second += h*(state[i].first + state[i].second);
  state[i].first += h;
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
  const uint32_t N = 10000;	// Number of initial conditions

  // State on host
  t_y state[N];

  for(uint32_t i = 0; i < N; ++i) {
    state[i].first = 0;
    state[i].second = i;
  }

  // Allocate statevector on device
  t_y* d_state = nullptr;
  check_cuda(cudaMalloc(&d_state, N * sizeof(t_y)), "cudaMalloc state");

  // Initialize statevector to something simple on device
  check_cuda(cudaMemcpy(d_state, state, N * sizeof(t_y),   cudaMemcpyHostToDevice), "copy to device");

  // Apply kernel
  const int threads = 256;
  const int blocks = (N + threads - 1) / threads;

  apply_rk4<<<blocks, threads>>>(d_state, N);
  check_cuda(cudaGetLastError(), "kernel launch");
  check_cuda(cudaDeviceSynchronize(), "kernel sync");

  // Copy back a few entries to sanity-check
  check_cuda(cudaMemcpy(state, d_state, sizeof(double), cudaMemcpyDeviceToHost), "copy to host");

  for(uint32_t i=0; i < 10; ++i) {
    printf("i=%d, ti=%g, yi=%g\n", i, state[i].first, state[i].second);
  }

  cudaFree(d_state);
  return 0;
}
