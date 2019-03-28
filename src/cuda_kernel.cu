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
__global__ void jacobi_kernel(double *U, double *Unew, double *A, double *F, int N){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	//for(int i=0;i<N;i++){
	double sigma = 0.0;

	if(i<N && y==0){
		for(int j=0; j<N; j++){
			if(j!=i)
				sigma+=A[j+i*N]*U[j];
		}
		Unew[i] = (F[i]-sigma)/A[i+i*N];
	}
	//}
}
//JACOBI SMOOTHING EXECUTED ON HOST/DEVICE TO MEASURE ERRORS. CALLED WITH KERNEL=TRUE FOR GPU LAUNCH
double jacobi_smoother(double *U, double *Unew, double *A, double *F, int N, dim3 blocksPerGrid, dim3 threadsPerBlock, bool KERNEL){
	int MAX_ITER = 1000;
	int iter=0;
	double *temp;
	//runs through jacobi naively, swaps pointers, iterates
	while(iter<MAX_ITER){
		if(KERNEL){
			//printf("%d %d\n", U, Unew);
			jacobi_kernel<<<blocksPerGrid, threadsPerBlock>>>(U, Unew, A, F, N);
		}else{
			printf("%d %d\n", U, Unew);
			
			for(int i=0;i<N;i++){
				double sigma = 0.0;
				for(int j=0; j<N; j++){
					if(j!=i)
						sigma+=A[j+i*N]*U[j];
				}
				Unew[i]= (F[i]-sigma)/A[i+i*N];
			}

		}
		temp = U;
		U = Unew;
		Unew = temp;
		iter++;
	}
	if(KERNEL)
		return 0.0;
	double jacobi_error =0.0;
	for(int i=0;i<N;i++){
		jacobi_error+=U[i];
		//printf("%lf ", U[i]);
	}
	//printf("\n");
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
			A[i*k+j]=2.0*i+1.0*j+1.0;
	for(int i=0;i<k;i++)
		for(int j=0;j<m;j++)
			B[i*m+j]=2.0*i+1.0*j+1.0;
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
	gpuErrchk(cudaEventRecord(gemm_start));
	gemm_kernel<<<blocksPerGrid,threadsPerBlock>>>(A_d, B_d, C_d, n, k, m);
	gpuErrchk(cudaEventRecord(gemm_stop));
	gpuErrchk(cudaMemcpy(C, C_d, n*m*sizeof(double), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaEventSynchronize(gemm_stop));
	
	//CALCULATE ERROR
	for(int i =0;i<n;i++)
		for(int j=0; j<m; j++)
			gemm_checksum-=C[i*m+j];
	
	cout<<"GEMM\n"<<"Error "<<gemm_checksum<<endl;
	gpuErrchk(cudaEventElapsedTime(&records[iteration].gemm_time, gemm_start, gemm_stop));
	cout<<"Time "<< records[iteration].gemm_time/1e3<< " Seconds"<<endl;
	cudaFree(A_d);
	cudaFree(B_d);
	cudaFree(C_d);
	free(A), free(B), free(C);

	//SYNCHRONIZE BETWEEN KENRNELS? WILL IMPLEMENT MULTIPLE STREAMS
	//gpuErrchk(cudaDeviceSynchronize());

	//JACOBI
	//INITIALIZE VECTORS U AND F,  MATRIX J , JxU=F (nxm)x(mx1)=(nx1)
	double *U, *F, *Unew,*U_d, *F_d, *Unew_d, *J_d;
	U = (double *)malloc(m*sizeof(double));
	Unew = (double *)malloc(m*sizeof(double));
	F = (double *)malloc(n*sizeof(double));
	for(int i=0; i<m; i++){
		U[i]=0.0;
		Unew[i]=0.0;
		F[i]=1.0;
	}
	double *J=(double *)malloc(n*m*sizeof(double));
	for(int i=0;i<n;i++){
		for(int j=0;j<n;j++){
			if(j==i-1 || j==i+1)
				J[i*n+j]=-1.0;
			else if(j==i)
				J[i*n+j]=2.0;
			else
				J[i*n+j]=0.0;
		}
	}
	//HOST SOLUTION
	double jacobi_checksum = jacobi_smoother(U, Unew, J, F, m, blocksPerGrid, threadsPerBlock, false);
	
	gpuErrchk(cudaMalloc(&U_d, m*sizeof(double)));
	gpuErrchk(cudaMalloc(&Unew_d, m*sizeof(double)));
	gpuErrchk(cudaMemcpy(U_d, U, m*sizeof(double), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(Unew_d, Unew, m*sizeof(double), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMalloc(&F_d, n*sizeof(double)));
	gpuErrchk(cudaMemcpy(F_d, F, n*sizeof(double), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMalloc(&J_d, n*m*sizeof(double)));
	cudaMemcpy(J_d, J, n*m*sizeof(double), cudaMemcpyHostToDevice); 

	//KERNEL, TIME, MEMCPY
	cudaEvent_t jacobi_start, jacobi_stop;
	gpuErrchk(cudaEventCreate(&jacobi_start));
	gpuErrchk(cudaEventCreate(&jacobi_stop));
	gpuErrchk(cudaEventRecord(jacobi_start));
	//TRIED DEBUGING BY EXPLICITLY SWAPPING POINTERS COMMENT jacobi_smoother() BELOW TO DEBUG
	/*for(int k=0;k<1000; k++){
		if(k%2==0)
			jacobi_kernel<<<blocksPerGrid, threadsPerBlock>>>(U_d, Unew_d, J_d, F_d, m);
		else
			jacobi_kernel<<<blocksPerGrid, threadsPerBlock>>>(Unew_d, U_d, J_d, F_d, m);
	}*/
	jacobi_smoother(U_d, Unew_d, J_d, F_d, m, blocksPerGrid, threadsPerBlock, true);
	gpuErrchk(cudaEventRecord(jacobi_stop));
	gpuErrchk(cudaMemcpy(U, U_d, m*sizeof(double), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaEventSynchronize(jacobi_stop));
	
	//UPDATE CHECKSUM WITH GPU SOLUTION
	for(int i=0;i<m;i++){
		printf("%lf ", U[i]);
		jacobi_checksum-=U[i];
	}
	printf("\n");
	cout<<"JACOBI\n"<<"Error "<<jacobi_checksum<<endl;
	gpuErrchk(cudaEventElapsedTime(&records[iteration].jacobi_time, jacobi_start, jacobi_stop));
	cout<<"Time "<< records[iteration].jacobi_time/1e3<< " Seconds"<<endl;
	

	cudaFree(U_d), cudaFree(Unew_d), cudaFree(F_d), cudaFree(J_d);
	free(U), free(Unew), free(F), free(J);
}

