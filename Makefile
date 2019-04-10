all: tuner 

NVCC := nvcc
CC:= $(NVCC)
HEADIR := src
CUDADIR := $(CUDA_HOME)
NVCCFLAGS := -Xcompiler "-fopenmp -O2"
CPPFLAGS := --std=c++11 -I$(CUDADIR)/include -I$(HEADIR)
LDFLAGS := -L$(CUDADIR)/lib
LDLIBS := -lcublas
OBJECTS :=cuda_kernel.o tuner

tuner: gpu_tuner.cu cuda_kernel.o
	$(CC) $(NVCCFLAGS) $(LDFLAGS) $(LDLIBS) $(CPPFLAGS) $^ -o $@

cuda_kernel.o: src/cuda_kernel.cu
	$(NVCC) $(LDFLAGS) $(LDLIBS) $(CPPFLAGS) -c $^


.PHONY: clean
clean:
	-rm $(OBJECTS)
