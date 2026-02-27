CFLAGS=-g -ggdb3 -Wall -Wextra -pedantic
LDFLAGS=-lgmp
NVCCFLAGS=-O3 -std=c++17 -ccbin /usr/bin/g++-13

PROGS = rk4 rk4_no_cuda rk4_no_cuda_omp rk4_mpi

all: $(PROGS)


# Add switched -g -G for debugging with cuda-gdb
rk4: rk4.cu
	nvcc $(NVCCFLAGS) $< -o $@

rk4_no_cuda: rk4_no_cuda.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

rk4_no_cuda_omp: rk4_no_cuda_omp.cpp
	$(CXX) -fopenmp $(CXXFLAGS) $< -o $@

# Run MPI version like this:
# mpirun -np 4 ./rk4_mpi
rk4_mpi: rk4_mpi.cpp
	mpic++ -O3 -std=c++17 rk4_mpi.cpp -o rk4_mpi

run: $(PROGS)
	echo '----------------------------------------------------------------'
	time python ./rk4.py >/dev/null
	echo '----------------------------------------------------------------'
	time ./rk4_no_cuda >/dev/null
	echo '----------------------------------------------------------------'
	time ./rk4_no_cuda_omp >/dev/null
	echo '----------------------------------------------------------------'
	time mpirun -np 10 ./rk4_mpi >/dev/null
	echo '----------------------------------------------------------------'
	time ./rk4 >/dev/null


clean:
	rm $(PROGS)
