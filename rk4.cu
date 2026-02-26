#include <cuda_runtime.h>
#include <cuComplex.h>
#include <stdint.h>
#include <stdio.h>


typedef std::pair<double, double> t_y;

__device__
float f(float t, float y) {
  return t + y;		// Example ODE: dy/dt = -2y
}

// Exact solution y = %e^t*y0+%e^t-t-1

float y_exact(float t, float y0) {
  return std::exp(t)*y0+exp(t)-t-1;
}


__global__
void rk4_kernel(float* y_results, float* initial_conditions, float h, int steps, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    float y = initial_conditions[idx];
    float t = 0.0f;

    for (int i = 0; i < steps; i++) {
      float k1 = h * f(t, y);
      float k2 = h * f(t + 0.5f * h, y + 0.5f * k1);
      float k3 = h * f(t + 0.5f * h, y + 0.5f * k2);
      float k4 = h * f(t + h, y + k3);

      y = y + (k1 + 2.0f * k2 + 2.0f * k3 + k4) / 6.0f;
      t += h;
    }
    y_results[idx] = y; // Store final result back to global memory
  }
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

  float y[N], y0[N];		// Memory on host

  const double h=.1;
  const int n_steps = std::floor(1/h);

  float* y_results = nullptr; float* initial_conditions = nullptr;
  check_cuda(cudaMalloc(&y_results, N * sizeof(float)), "cudaMalloc y results");  
  check_cuda(cudaMalloc(&initial_conditions, N * sizeof(float)), "cudaMalloc initial conds");




  // Initialize init conditions
  for(uint32_t i=0; i < N; ++i)
    y0[i] = i;


  // Copy to device
  check_cuda(cudaMemcpy(initial_conditions, y0, N*sizeof(float), cudaMemcpyHostToDevice), "copy to device");


  // Apply kernel
  const int threads = 256;
  const int blocks = (N + threads - 1) / threads;


  for (uint32_t i=0; i < 1/h; ++i) {
    rk4_kernel<<<blocks, threads>>>(y_results, initial_conditions, h, n_steps, N);
  }
  check_cuda(cudaGetLastError(), "kernel launch");
  check_cuda(cudaDeviceSynchronize(), "kernel sync");

  // Copy back a few entries to sanity-check
  check_cuda(cudaMemcpy(y, y_results, N*sizeof(float), cudaMemcpyDeviceToHost), "copy to host");

  for(uint32_t i=0; i < 100; ++i) {
    printf("i=%d, y0=%g, yi=%g, yi_exact=%g, err=%g\n",
	   i, y0[i], y[i], y_exact(1, y0[i]), y_exact(1,y0[i]) - y[i]);
  }

  cudaFree(y_results);
  cudaFree(initial_conditions);  
  return 0;
}
