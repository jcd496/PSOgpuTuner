#include <cublas_v2.h>
#include <iostream>
#include "param_struct.hpp"

using namespace std;
__global__ void test_kernel(int size, double * array_x, double * array_y){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i<size)
		array_y[i]=array_y[i]+array_x[i];
}

//PSO will call kernel_wrapper with different parameters and kernel_wrapper will evaluate kernel and store statistics
void kernel_wrapper(int iteration, dim3 blocksPerGrid, dim3 threadsPerBlock, record_t * records){

	//ARRAY OF STRUCTURES OR STRUCTURE OF ARRAYS? ARRAY OF STRUCTURES SEEMS TO MAKE MORE SENSE
	//CLEANER CODE AND THE WHOLE STRUCTURE WILL BE ACCESSED SEQUENTIALLY, NOT AN INTERNAL ARRAY.
	records[iteration].parameters.threads_per_block = threadsPerBlock;
	records[iteration].parameters.blocks_per_grid = blocksPerGrid;	
	size_t size = 1000000;
	double * array = (double *)malloc(size*sizeof(double));
	for(int i=0;i<size;i++) array[i]=1.0;
	
	double *cu_array_x, *cu_array_y;
	
	cudaMalloc(&cu_array_x, size*sizeof(double));
	cudaMemcpy(cu_array_x, array, size*sizeof(double), cudaMemcpyHostToDevice); 
	cudaMalloc(&cu_array_y, size*sizeof(double));
	cudaMemcpy(cu_array_y, array, size*sizeof(double), cudaMemcpyHostToDevice); 
	
	cudaEvent_t gemm_start, gemm_stop;
	cudaEventCreate(&gemm_start), cudaEventCreate(&gemm_stop);
	dim3 blocks(4096), threads(256);
	cudaEventRecord(gemm_start,0);
	test_kernel<<<4096,256>>>(size, cu_array_x, cu_array_y);
	cudaEventRecord(gemm_stop,0);
	cudaMemcpy(array, cu_array_y, size*sizeof(double), cudaMemcpyDeviceToHost);
	cudaEventSynchronize(gemm_stop);
	
	//for(int i =0;i<size;i++) cout<<array[i]<<endl;
	float time;
	cudaEventElapsedTime(&time, gemm_start, gemm_stop);
	cout<<time<<endl;
	
	cudaFree(cu_array_x);
	cudaFree(cu_array_y);
	free(array);

}


void gemm_kernel(cublasHandle_t handle,const float *A, const float *B, float *C, const int m, const int k, const int n){

	


}
