#include "cublas_v2.h"
#include "particle_struct.hpp"
#ifndef CUDA_KERNELS
#define CUDA_KERNELS
void kernel_wrapper(int iteration, dim3 blocksPerGrid, dim3 threadsPerBlock, particle_t * particles, int problem_size);
//void gemm_kernel(const float *A, const float *B, float *C, const int n, const int k, const int m);





#endif
