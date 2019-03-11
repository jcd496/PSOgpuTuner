all: tuner 

NVCC := nvcc
CC:= $(NVCC)
HEADIR := src
CUDADIR := $(CUDA_HOME)
CPPFLAGS := -O2 --std=c++11 -I$(CUDADIR)/include -I$(HEADIR)
LDFLAGS := -L$(CUDADIR)/lib
LDLIBS := -lcublas
OBJECTS :=cuda_kernel.o tuner

tuner: gpu_tuner.cpp cuda_kernel.o
	$(CC) $(LDFLAGS) $(LDLIBS) $(CPPFLAGS) $^ -o $@

cuda_kernel.o: src/cuda_kernel.cu
	$(NVCC) -c $^

.PHONY: clean
clean:
	-rm $(OBJECTS)
