#include "particle_struct.hpp"
#ifndef CUDA_KERNELS
#define CUDA_KERNELS
void kernel_wrapper(int iteration, dim3 blocksPerGrid, dim3 threadsPerBlock, particle_t * particles, int problem_size);
__global__ void dummy_kernel(int n);

#endif
