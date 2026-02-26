CFLAGS=-g -ggdb3 -Wall -Wextra -pedantic
LDFLAGS=-lgmp
NVCCFLAGS=-O3 -std=c++17 -ccbin /usr/bin/g++-13

PROGS = rk4 rk4_no_cuda rk4_no_cuda_omp

all: $(PROGS)


# Add switched -g -G for debugging with cuda-gdb
rk4: rk4.cu
	nvcc $(NVCCFLAGS) $< -o $@

rk4_no_cuda: rk4_no_cuda.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

rk4_no_cuda_omp: rk4_no_cuda_omp.cpp
	$(CXX) -omp $(CXXFLAGS) $< -o $@


clean:
	rm $(PROGS)
