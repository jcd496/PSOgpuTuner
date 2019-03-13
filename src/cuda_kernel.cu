#include <cublas_v2.h>
#include <iostream>
#include "param_struct.hpp"

using namespace std;
__global__ void test_kernel(int size, double * array_x, double * array_y){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i<size)
		for(int i=0;i<size;i++)
			array_y[i]=array_y[i]+array_x[i];
}
__global__ void gemm_kernel(const double *A, const double *B, double *C, const int n, const int k, const int m){
	int col =  blockIdx.x*blockDim.x + threadIdx.x;
	int row =  blockIdx.y*blockDim.y + threadIdx.y;
	if(row<n && col<m)
		for(int p=0; p<k; p++)
			C[row*m+col] = C[row*m+col] + A[row*k+p]*B[p*k+col];

}


//PSO will call kernel_wrapper with different parameters and kernel_wrapper will evaluate kernel and store statistics
void kernel_wrapper(int iteration, dim3 blocksPerGrid, dim3 threadsPerBlock, record_t * records){

	//ARRAY OF STRUCTURES OR STRUCTURE OF ARRAYS? ARRAY OF STRUCTURES SEEMS TO MAKE MORE SENSE
	//CLEANER CODE AND THE WHOLE STRUCTURE WILL BE ACCESSED SEQUENTIALLY, NOT AN INTERNAL ARRAY.
	records[iteration].parameters.threads_per_block = threadsPerBlock;
	records[iteration].parameters.blocks_per_grid = blocksPerGrid;	

	//INITIALIZE GEMM MATRICIES
	size_t n, k, m;
	n = k = m = 32;
	double *A, *B, *C;
	A = (double *)malloc(n*k*sizeof(double)), B= (double *)malloc(k*m*sizeof(double)), C=(double *)malloc(n*m*sizeof(double));
	for(int i=0;i<n;i++)
		for(int j=0;j<k;j++)
			A[i*k+j]=2.0*i+1.0*j;
	for(int i=0;i<k;i++)
		for(int j=0;j<m;j++)
			B[i*m+j]=2.0*i+1.0*j;
	for(int i=0;i<n;i++)
		for(int j=0;j<m;j++)
			C[i*m+j]=0.0;
	double checksum = 0.0;
	for(int i=0;i<n;i++)
		for(int p=0;p<k;p++)
			for(int j=0;j<m;j++)
				checksum+=A[i*k+p]*B[p*k+j];
	
	//DECLARE DEVICE POINTERS, CUDAMALLOC,  AND COPY MEMORY
	double *A_d, *B_d, *C_d;
	cudaMalloc(&A_d, n*k*sizeof(double));
	cudaMemcpy(A_d, A, n*k*sizeof(double), cudaMemcpyHostToDevice); 
	cudaMalloc(&B_d, k*m*sizeof(double));
	cudaMemcpy(B_d, B, k*m*sizeof(double), cudaMemcpyHostToDevice); 
	cudaMalloc(&C_d, n*m*sizeof(double));
	cudaMemcpy(C_d, C, n*m*sizeof(double), cudaMemcpyHostToDevice); 
	//LAUNCH KERNEL, RECORD TIME, COPY KERNEL RESULTS TO HOST
	cudaEvent_t gemm_start, gemm_stop;
	cudaEventCreate(&gemm_start), cudaEventCreate(&gemm_stop);
	cudaEventRecord(gemm_start,0);
	gemm_kernel<<<blocksPerGrid,threadsPerBlock>>>(A_d, B_d, C_d, n, k, m);
	cudaEventRecord(gemm_stop,0);
	cudaMemcpy(C, C_d, n*m*sizeof(double), cudaMemcpyDeviceToHost);
	cudaEventSynchronize(gemm_stop);
	
	double sum = 0.0;
	for(int i =0;i<n;i++)
		for(int j=0; j<m; j++)
			sum+=C[i*m+j];
	
	cout<<"GEMM\n"<<"Error "<<checksum-sum<<endl;
	cudaEventElapsedTime(&records[iteration].gemm, gemm_start, gemm_stop);
	cout<<"time "<< records[iteration].gemm/1e3<< "seconds"<<endl;
	cudaFree(A_d);
	cudaFree(B_d);
	cudaFree(C_d);
	free(A), free(B), free(C);

}



