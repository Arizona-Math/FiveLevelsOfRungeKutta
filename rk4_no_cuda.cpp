#include <stdint.h>
#include <stdio.h>
#include <cmath>

inline
float f(float t, float y) {
  return t + y;		// Example ODE: dy/dt = -2y
}

// Exact solution y = %e^t*y0+%e^t-t-1
float y_exact(float t, float y0) {
  return std::exp(t)*y0+exp(t)-t-1;
}


void rk4_kernel(float* y_results, float* initial_conditions, float h, int steps, int N) {
  for(uint32_t idx = 0; idx < N; ++idx) {
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


int main()
{
  const uint32_t N = 10000000;	// Number of initial conditions (ten million)

  float *y = new float[N];
  float *y0 = new float[N];

  const float h=.1;
  const int n_steps = std::floor(1/h);


  // Initialize init conditions
  for(uint32_t i=0; i < N; ++i)
    y0[i] = i;



  rk4_kernel(y, y0, h, n_steps, N);

  for(uint32_t i=0; i < 100; ++i) {
    printf("i=%d, y0=%g, yi=%g, yi_exact=%g, err=%g\n",
	   i, y0[i], y[i], y_exact(1, y0[i]), y_exact(1,y0[i]) - y[i]);
  }

  delete[] y;
  delete[] y0;
  return 0;
}
