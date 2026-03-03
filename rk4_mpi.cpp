// rk4_mpi.cpp
#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

// ODE RHS: y' = t + y  (matches your original code)
inline float f(float t, float y) {
  return t + y;
}

// Exact solution for y' = t + y with y(0)=y0:
// y(t) = e^t*y0 + e^t - t - 1
inline float y_exact(float t, float y0) {
  float et = std::exp(t);
  return et * y0 + et - t - 1.0f;
}

// Determine this rank's global index range [start, start+local_N)
static inline void block_decompose(std::uint64_t N, int rank, int size,
                                   std::uint64_t &start, std::uint64_t &local_N) {
  std::uint64_t base = N / (std::uint64_t)size;
  std::uint64_t rem  = N % (std::uint64_t)size;

  local_N = base + (rank < (int)rem ? 1 : 0);
  start   = (std::uint64_t)rank * base + (std::uint64_t)std::min(rank, (int)rem);
}

struct Record {
  int   i;   // global index
  float y0;
  float y;
};

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Match the original intent: ten million initial conditions
  const std::uint64_t N = 10'000'000ULL;

  const float h = 0.1f;
  const int n_steps = (int)std::floor(1.0f / h);

  // Local chunk
  std::uint64_t start = 0, local_N = 0;
  block_decompose(N, rank, size, start, local_N);

  std::vector<float> local_y;
  local_y.resize((size_t)local_N);

  // Compute RK4 for each local initial condition y0 = global_index
  for (std::uint64_t j = 0; j < local_N; ++j) {
    float y = (float)(start + j);
    float t = 0.0f;

    for (int step = 0; step < n_steps; ++step) {
      float k1 = h * f(t, y);
      float k2 = h * f(t + 0.5f * h, y + 0.5f * k1);
      float k3 = h * f(t + 0.5f * h, y + 0.5f * k2);
      float k4 = h * f(t + h,       y + k3);

      y = y + (k1 + 2.0f * k2 + 2.0f * k3 + k4) / 6.0f;
      t += h;
    }

    local_y[(size_t)j] = y;
  }

  // We will print only the first 100 entries like your original code.
  // Instead of gathering all 10M results to rank 0, each rank sends just
  // the records it owns among indices [0,100).
  const int PRINT_K = 100;

  std::vector<Record> sendrecs;
  {
    std::uint64_t end = start + local_N;
    int lo = (int)std::max<std::uint64_t>(0ULL, start);
    int hi = (int)std::min<std::uint64_t>((std::uint64_t)PRINT_K, end);

    // If this rank owns any of i=0..99
    if (hi > lo) {
      sendrecs.reserve((size_t)(hi - lo));
      for (int i = lo; i < hi; ++i) {
        std::uint64_t j = (std::uint64_t)i - start;
        Record r;
        r.i  = i;
        r.y0 = (float)i;
        r.y  = local_y[(size_t)j];
        sendrecs.push_back(r);
      }
    }
  }

  // Gather counts to rank 0
  int sendcount_bytes = (int)(sendrecs.size() * sizeof(Record));
  std::vector<int> recvcounts_bytes, displs_bytes;
  int total_bytes = 0;

  if (rank == 0) {
    recvcounts_bytes.resize((size_t)size, 0);
  }

  MPI_Gather(&sendcount_bytes, 1, MPI_INT,
             rank == 0 ? recvcounts_bytes.data() : nullptr, 1, MPI_INT,
             0, MPI_COMM_WORLD);

  std::vector<unsigned char> recvbuf;
  if (rank == 0) {
    displs_bytes.resize((size_t)size, 0);
    for (int r = 0; r < size; ++r) {
      displs_bytes[(size_t)r] = total_bytes;
      total_bytes += recvcounts_bytes[(size_t)r];
    }
    recvbuf.resize((size_t)total_bytes);
  }

  // Gatherv raw bytes
  MPI_Gatherv(
      (void*)sendrecs.data(), sendcount_bytes, MPI_BYTE,
      rank == 0 ? (void*)recvbuf.data() : nullptr,
      rank == 0 ? recvcounts_bytes.data() : nullptr,
      rank == 0 ? displs_bytes.data() : nullptr,
      MPI_BYTE, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    // Reconstruct first 100 (y0 and y). There should be exactly one record per i.
    float y0_print[PRINT_K];
    float y_print[PRINT_K];
    for (int i = 0; i < PRINT_K; ++i) {
      y0_print[i] = (float)i;
      y_print[i]  = NAN;
    }

    // Unpack records
    size_t nrecs = recvbuf.size() / sizeof(Record);
    const Record* recs = (const Record*)recvbuf.data();
    for (size_t k = 0; k < nrecs; ++k) {
      int i = recs[k].i;
      if (0 <= i && i < PRINT_K) {
        y0_print[i] = recs[k].y0;
        y_print[i]  = recs[k].y;
      }
    }

    // Print results
    for (int i = 0; i < PRINT_K; ++i) {
      float yi_exact = y_exact(1.0f, y0_print[i]);
      float err = yi_exact - y_print[i];
      std::printf("i=%d, y0=%g, yi=%g, yi_exact=%g, err=%g\n",
                  i, y0_print[i], y_print[i], yi_exact, err);
    }
  }

  MPI_Finalize();
  return 0;
}
