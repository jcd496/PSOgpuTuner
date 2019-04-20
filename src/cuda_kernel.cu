#include <cublas_v2.h>
#include <iostream>
#include <float.h>
#include "particle_struct.hpp"
#include "cuda_kernel.hpp"

//ERROR HANDLING FOR GPU CALLS (TAKEN FROM STACK OVERFLOW)
#define gpuErrchk(ans) {gpuAssert((ans),  __FILE__, __LINE__);}
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
	if(code!=cudaSuccess){
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if(abort) exit(code);
	}
}
using namespace std;
//GEMM KERNEL, NO OPTIMIZATIONS, MAY INCLUDE TILING ETC
__global__ void gemm_kernel(const float *A, const float *B, float *C, const int n, const int k, const int m){
	int col =  blockIdx.x*blockDim.x + threadIdx.x;
	int row =  blockIdx.y*blockDim.y + threadIdx.y;
	if(row<n && col<m){
		float element = 0.0;
		for(int p=0; p<k; p++)
			element += A[row*k+p]*B[p*k+col];
		C[row*m+col] = element + 0.0;
	}
}
__global__ void jacobi_kernel(double *U, double *Unew, double *A, double *F, int N){
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	//for(int i=0;i<N;i++){
	double sigma = 0.0;

	if(row<N && col==0){
		for(int j=0; j<N; j++){
			if(j!=row)
				sigma+=A[j+row*N]*U[j];
		}
		Unew[row] = (F[row]-sigma)/A[row+row*N];
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
			jacobi_kernel<<<blocksPerGrid, threadsPerBlock>>>(U, Unew, A, F, N);
		}else{
			
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
	for(int i=0;i<N;i++) jacobi_error+=U[i];
	return jacobi_error;
}
double jacobi_host_solver(int problem_size){
	int m, n;
	m=n=problem_size;
	//JACOBI
	//INITIALIZE VECTORS U AND F,  MATRIX J , JxU=F (nxm)x(mx1)=(nx1)
	double *U, *F, *Unew;
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
	dim3 dummy1(1);
	dim3 dummy2(1);
	double jacobi_checksum = jacobi_smoother(U, Unew, J, F, m, dummy1, dummy2, false);
	free(U), free(Unew), free(F), free(J);
	return jacobi_checksum;
}
void mem_to_device(device_pointers_t * pointers, int problem_size){
	//INITIALIZE GEMM MATRICIES
	int n, k, m;
	n = k = m = problem_size;
	float *A, *B, *C;
	A = (float *)malloc(n*k*sizeof(float)), B= (float *)malloc(k*m*sizeof(float)), C=(float *)malloc(n*m*sizeof(float));
	for(int i=0;i<n;i++)
		for(int j=0;j<k;j++)
			A[i*k+j]=1.0;
	for(int i=0;i<k;i++)
		for(int j=0;j<m;j++)
			B[i*m+j]=1.0;
	for(int i=0;i<n;i++)
		for(int j=0;j<m;j++)
			C[i*m+j]=0.0;

	//DECLARE DEVICE POINTERS, CUDAMALLOC,  AND COPY MEMORY
	float *A_d, *B_d, *C_d;
	gpuErrchk(cudaMalloc(&A_d, n*k*sizeof(float)));
	gpuErrchk(cudaMemcpy(A_d, A, n*k*sizeof(float), cudaMemcpyHostToDevice)); 
	gpuErrchk(cudaMalloc(&B_d, k*m*sizeof(float)));
	gpuErrchk(cudaMemcpy(B_d, B, k*m*sizeof(float), cudaMemcpyHostToDevice)); 
	gpuErrchk(cudaMalloc(&C_d, n*m*sizeof(float)));
	gpuErrchk(cudaMemcpy(C_d, C, n*m*sizeof(float), cudaMemcpyHostToDevice)); 
	//HOST SOLUTION	
	for(int i=0;i<n;i++)
		for(int p=0;p<k;p++)
			for(int j=0;j<m;j++)
				pointers->gemm_checksum+=A[i*k+p]*B[p*k+j];

	pointers->A = A_d, pointers->B = B_d, pointers->C = C_d;
	free(A), free(B), free(C);
	//JACOBI
	//INITIALIZE VECTORS U AND F,  MATRIX J , JxU=F (nxm)x(mx1)=(nx1)
	double *U, *F, *Unew, *U_d, *F_d, *Unew_d, *J_d;
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
	gpuErrchk(cudaMalloc(&U_d, m*sizeof(double)));
	gpuErrchk(cudaMalloc(&Unew_d, m*sizeof(double)));
	gpuErrchk(cudaMemcpy(U_d, U, m*sizeof(double), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(Unew_d, Unew, m*sizeof(double), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMalloc(&F_d, n*sizeof(double)));
	gpuErrchk(cudaMemcpy(F_d, F, n*sizeof(double), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMalloc(&J_d, n*m*sizeof(double)));
	gpuErrchk(cudaMemcpy(J_d, J, n*m*sizeof(double), cudaMemcpyHostToDevice)); 
	
	pointers->U = U_d, pointers->Unew = Unew_d, pointers->F = F_d, pointers->J = J_d;
	free(U), free(Unew), free(F), free(J);
}

void free_device(device_pointers_t * pointers){
	cudaFree(pointers->A);
	cudaFree(pointers->B);
	cudaFree(pointers->C);
	cudaFree(pointers->U);
	cudaFree(pointers->Unew);
	cudaFree(pointers->F);
	cudaFree(pointers->J);
}

void reset_device_mem(double * U_d, double * Unew_d, int m){
	double *U = (double *)malloc(m*sizeof(double));
	double *Unew = (double *)malloc(m*sizeof(double));
	for(int i=0; i<m; i++){
		U[i] = 0.0;
		Unew[i] = 0.0;
	}
	gpuErrchk(cudaMemcpy(U_d, U, m*sizeof(double), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(Unew_d, Unew, m*sizeof(double), cudaMemcpyHostToDevice));
	free(U), free(Unew);	

}

//PSO will call kernel_wrapper with different parameters and kernel_wrapper will evaluate kernel and store statistics
void kernel_wrapper(int id, dim3 blocksPerGrid, dim3 threadsPerBlock, particle_t * particles, int problem_size, device_pointers_t * pointers, double jacobi_host_solution){
	int n, k, m;
	n = k = m = problem_size;
	
	float *A_d, *B_d, *C_d;
	A_d = pointers->A;
	B_d = pointers->B;
	C_d = pointers->C;
	
	float *C = (float *)malloc(n*m*sizeof(float));

	//LAUNCH KERNEL, RECORD TIME, COPY KERNEL RESULTS TO HOST
	cudaEvent_t gemm_start, gemm_stop;
	gpuErrchk(cudaEventCreate(&gemm_start));
	gpuErrchk(cudaEventCreate(&gemm_stop));
	gpuErrchk(cudaEventRecord(gemm_start));
	gemm_kernel<<<blocksPerGrid,threadsPerBlock>>>(A_d, B_d, C_d, n, k, m);
	gpuErrchk(cudaEventRecord(gemm_stop));
	gpuErrchk(cudaMemcpy(C, C_d, n*m*sizeof(float), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaEventSynchronize(gemm_stop));
	
	//CALCULATE ERROR
	double gemm_checksum = pointers->gemm_checksum;
	for(int i =0;i<n;i++)
		for(int j=0; j<m; j++)
			gemm_checksum-=C[i*m+j];
    free(C);

	//cout<<"GEMM\n"<<"Error "<<gemm_checksum<<endl;
	gpuErrchk(cudaEventElapsedTime(&particles[id].gemm_time, gemm_start, gemm_stop));
	//cout<<"Time "<< particles[id].gemm_time/1e3<< " Seconds"<<endl;


	//JACOBI
	double *U_d, *Unew_d, *F_d, *J_d;
	U_d = pointers->U, Unew_d = pointers->Unew;
	F_d = pointers->F, J_d = pointers->J;

	double *U = (double*)malloc(m*sizeof(double));
	//KERNEL, TIME, MEMCPY
	cudaEvent_t jacobi_start, jacobi_stop;
	gpuErrchk(cudaEventCreate(&jacobi_start));
	gpuErrchk(cudaEventCreate(&jacobi_stop));
	gpuErrchk(cudaEventRecord(jacobi_start));
	jacobi_smoother(U_d, Unew_d, J_d, F_d, m, blocksPerGrid, threadsPerBlock, true);
	gpuErrchk(cudaEventRecord(jacobi_stop));
	gpuErrchk(cudaMemcpy(U, U_d, m*sizeof(double), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaEventSynchronize(jacobi_stop));
	
	reset_device_mem(pointers->U, pointers->Unew, m);

	//UPDATE CHECKSUM WITH GPU SOLUTION
	double jacobi_checksum = 0.0;
	for(int i=0;i<m;i++){
		jacobi_checksum-=U[i];
	}
	free(U);
	jacobi_checksum += jacobi_host_solution;//jacobi_host_solver(problem_size);

	//cout<<"JACOBI\n"<<"Error "<<jacobi_checksum<<endl;
	gpuErrchk(cudaEventElapsedTime(&particles[id].jacobi_time, jacobi_start, jacobi_stop));
	//cout<<"Time "<< particles[id].jacobi_time/1e3<< " Seconds"<<endl;

	particles[id].total_time = particles[id].gemm_time + particles[id].jacobi_time;
	if(jacobi_checksum || gemm_checksum){
		fprintf(stderr, "GPU Mem Error at grid %d %d, thread %d %d\n", blocksPerGrid.x, blocksPerGrid.y, threadsPerBlock.x, threadsPerBlock.y);
		particles[id].total_time=FLT_MAX;
	}	 
}
//dummy kernel to establish connection with device
__global__ void dummy_kernel(int n){
	int idx = blockIdx.x*blockDim.x +threadIdx.x;
	if(idx<n)
		n = idx + n;
}
