#include "cublas_v2.h"
#ifndef CUDA_KERNELS
#define CUDA_KERNELS

void kernel_wrapper(int iteration, dim3 blocksPerGrid, dim3 threadsPerBlock, struct param_time_record * records);
void gemm_kernel(const float *A, const float *B, float *C, const int n, const int k, const int m);





#endif
