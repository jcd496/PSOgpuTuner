#include "particle_struct.hpp"
#ifndef CUDA_KERNELS
#define CUDA_KERNELS


typedef struct device_pointers{
	float * A;
	float * B;
	float * C;
	double * U;
	double * Unew;
	double * F;
	double * J;
	double gemm_checksum;
	double jacobi_checksum;
} device_pointers_t;
void mem_to_device(device_pointers_t * pointers, int problem_size);
void free_device(device_pointers_t *pointers);
void kernel_wrapper(int iteration, dim3 blocksPerGrid, dim3 threadsPerBlock, particle_t * particles, int problem_size, device_pointers_t * pointers, double jacobi_host_solution);
__global__ void dummy_kernel(int n);
double jacobi_host_solver(int problem_size);
#endif
