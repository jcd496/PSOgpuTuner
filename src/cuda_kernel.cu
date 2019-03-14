#include <cublas_v2.h>
#include <iostream>
#include "param_struct.hpp"
//ERROR HANDLING FOR GPU CALLS (TAKEN FROM STACK OVERFLOW)
#define gpuErrchk(ans) {gpuAssert((ans),  __FILE__, __LINE__);}
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
	if(code!=cudaSuccess){
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if(abort) exit(code);
	}
}
using namespace std;
//THIS IS JUST FOR EXPLORATION, WILL BE DELETED
__global__ void test_kernel(int size, double * array_x, double * array_y){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i<size)
		for(int i=0;i<size;i++)
			array_y[i]=array_y[i]+array_x[i];
}
//GEMM KERNEL, NO OPTIMIZATIONS, MAY INCLUDE TILING ETC
__global__ void gemm_kernel(const double *A, const double *B, double *C, const int n, const int k, const int m){
	int col =  blockIdx.x*blockDim.x + threadIdx.x;
	int row =  blockIdx.y*blockDim.y + threadIdx.y;
	if(row<n && col<m)
		for(int p=0; p<k; p++)
			C[row*m+col] = C[row*m+col] + A[row*k+p]*B[p*k+col];

}
//RESIDUAL ERROR USED IN JACOBI SMOOTHING
__host__ __device__ double residual(double *U, double *A ,double *F, int N){
	int i, j;
	double residual = 0.0;
	for(i=1;i<=N; i++){
		double element = 0.0;
		for(j=0;j<N;j++)
			element += A[j+i*N]*U[j];
		residual+=element*element;
	}
	return sqrtf(residual);
}
//JACOBI SMOOTHING KERNEL.  NO OPTIMIZATIONS, MAY INCLUDE TILING ETC
__global__ void jacobi_kernel(double *U, double *Unew, double *A, double *F, int N){
	//AxU=F
	double tolerance =1e-6;
	int MAX_ITER = 1000;
	double residual_original, residual_cur;
	residual_original = residual(U,A,F,N);
	residual_cur = residual_original;
	int iter;
	iter = 0;
	while(iter<MAX_ITER && residual_cur/residual_original > tolerance){
		//int j = blockIdx.x*blockDim.x + threadIdx.x; ONE DIMENSIONAL BLOCKS??
		int i = blockIdx.x*blockDim.x + threadIdx.x;
		//for(int i=0;i<N;i++){
			double sigma = 0.0;
			for(int j=0; j<N; j++){
				if(j!=i)
					sigma+=A[j+i*N]*U[j];
			}
			Unew[i]= 1/A[i+i*N]*(F[i]-sigma);
		//}
		__syncthreads();
		if(i==1)
			residual_cur = residual(U,A,F,N);
		__syncthreads();
		double * temp = U;
		U = Unew;
		Unew = temp;
		iter++;
	}
}
//JACOBI SMOOTHING EXECUTED ON HOST TO HANDLE ERRORS
double jacobi_host(double *U, double *A, double *F, int N){
	double tolerance =1e-6;
	int MAX_ITER = 1000;
	double residual_original, residual_cur;
	residual_original = residual(U,A,F,N);
	residual_cur = residual_original;
	int iter;
	iter = 0;
	double * Unew = (double *)malloc(N*sizeof(double));
	for(int i=0;i<N;i++) Unew[i]=0.0;
	while(iter<MAX_ITER && residual_cur/residual_original > tolerance){
		for(int i=0;i<N;i++){
			double sigma = 0.0;
			for(int j=0; j<N; j++){
				if(j!=i)
					sigma+=A[j+i*N]*U[j];
			}
			Unew[i]= 1/A[i+i*N]*(F[i]-sigma);
		}
		residual_cur = residual(U,A,F,N);
		double *temp = U;
		U = Unew;
		Unew = temp;
		iter++;
	}

	double jacobi_error =0.0;
	for(int i=0;i<N;i++) jacobi_error+=U[i];
	free(Unew);
	return jacobi_error;
}

//PSO will call kernel_wrapper with different parameters and kernel_wrapper will evaluate kernel and store statistics
void kernel_wrapper(int iteration, dim3 blocksPerGrid, dim3 threadsPerBlock, record_t * records){

	//ARRAY OF STRUCTURES OR STRUCTURE OF ARRAYS? ARRAY OF STRUCTURES SEEMS TO MAKE MORE SENSE
	//CLEANER CODE AND THE WHOLE STRUCTURE WILL BE ACCESSED SEQUENTIALLY, NOT AN INTERNAL ARRAY.
	records[iteration].parameters.threads_per_block = threadsPerBlock;
	records[iteration].parameters.blocks_per_grid = blocksPerGrid;	

	//INITIALIZE GEMM MATRICIES
	size_t n, k, m;
	n = k = m = 1024;
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
	double gemm_checksum = 0.0;
	for(int i=0;i<n;i++)
		for(int p=0;p<k;p++)
			for(int j=0;j<m;j++)
				gemm_checksum+=A[i*k+p]*B[p*k+j];
	
	//DECLARE DEVICE POINTERS, CUDAMALLOC,  AND COPY MEMORY
	double *A_d, *B_d, *C_d;
	gpuErrchk(cudaMalloc(&A_d, n*k*sizeof(double)));
	gpuErrchk(cudaMemcpy(A_d, A, n*k*sizeof(double), cudaMemcpyHostToDevice)); 
	gpuErrchk(cudaMalloc(&B_d, k*m*sizeof(double)));
	gpuErrchk(cudaMemcpy(B_d, B, k*m*sizeof(double), cudaMemcpyHostToDevice)); 
	gpuErrchk(cudaMalloc(&C_d, n*m*sizeof(double)));
	gpuErrchk(cudaMemcpy(C_d, C, n*m*sizeof(double), cudaMemcpyHostToDevice)); 
	//LAUNCH KERNEL, RECORD TIME, COPY KERNEL RESULTS TO HOST
	cudaEvent_t gemm_start, gemm_stop;
	gpuErrchk(cudaEventCreate(&gemm_start));
	gpuErrchk(cudaEventCreate(&gemm_stop));
	gpuErrchk(cudaEventRecord(gemm_start,0));
	gemm_kernel<<<blocksPerGrid,threadsPerBlock>>>(A_d, B_d, C_d, n, k, m);
	gpuErrchk(cudaEventRecord(gemm_stop,0));
	gpuErrchk(cudaMemcpy(C, C_d, n*m*sizeof(double), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaEventSynchronize(gemm_stop));
	//CALCULATE ERROR
	for(int i =0;i<n;i++)
		for(int j=0; j<m; j++)
			gemm_checksum-=C[i*m+j];
	
	cout<<"GEMM\n"<<"Error "<<gemm_checksum<<endl;
	gpuErrchk(cudaEventElapsedTime(&records[iteration].gemm_time, gemm_start, gemm_stop));
	cout<<"Time "<< records[iteration].gemm_time/1e3<< " Seconds"<<endl;
	
	//JACOBI
	//INITIALIZE VECTORS U AND F, C MATRIX WILL BE REUSED FROM GEMM, CxU=F (nxm)x(mx1)=(nx1)
	double *U, *F, *U_d, *F_d, *Unew_d;
	U = (double *)malloc(m*sizeof(double)), F = (double *)malloc(n*sizeof(double));
	for(int i=0; i<m; i++){
		U[i]=0.0;
		F[i]=1.0;
	}
	//cudaMemcpy(C_d, C, n*m*sizeof(double), cudaMemcpyHostToDevice); 
	gpuErrchk(cudaMalloc(&U_d, m*sizeof(double)));
	gpuErrchk(cudaMalloc(&Unew_d, m*sizeof(double)));
	gpuErrchk(cudaMemcpy(U_d, U, m*sizeof(double), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMalloc(&F_d, n*sizeof(double)));
	gpuErrchk(cudaMemcpy(F_d, F, n*sizeof(double), cudaMemcpyHostToDevice));
	//HOST SOLUTION
	double jacobi_checksum = jacobi_host(U, C, F, m);

	//KERNEL, TIME, MEMCPY
	cudaEvent_t jacobi_start, jacobi_stop;
	gpuErrchk(cudaEventCreate(&jacobi_start));
	gpuErrchk(cudaEventCreate(&jacobi_stop));
	gpuErrchk(cudaEventRecord(jacobi_start, 0));
	jacobi_kernel<<<1, 1>>>(U_d, Unew_d, C_d, F_d, m);
	//jacobi_kernel<<<blocksPerGrid, threadsPerBlock>>>(U, Unew_d, C, F, m);
	gpuErrchk(cudaEventRecord(jacobi_stop, 0));
	gpuErrchk(cudaMemcpy(U, U_d, m*sizeof(double), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaEventSynchronize(jacobi_stop));
	

	//UPDATE CHECKSUM WITH GPU SOLUTION
	for(int i=0;i<m;i++)
		jacobi_checksum-=U[i];


	cout<<"JACOBI\n"<<"Error "<<jacobi_checksum<<endl;
	gpuErrchk(cudaEventElapsedTime(&records[iteration].jacobi_time, jacobi_start, jacobi_stop));
	cout<<"Time "<< records[iteration].jacobi_time/1e3<< " Seconds"<<endl;
	



	cudaFree(A_d);
	cudaFree(B_d);
	cudaFree(C_d);
	free(A), free(B), free(C);
	free(U), free(F);
}



