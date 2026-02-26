CFLAGS=-g -ggdb3 -Wall -Wextra -pedantic
LDFLAGS=-lgmp

PROGS = rk4

all: $(PROGS)


# Add switched -g -G for debugging with cuda-gdb
rk4: rk4.cu
	nvcc -O3 -std=c++17 -ccbin /usr/bin/g++-13 $< -o $@


clean:
	rm $(PROGS)
