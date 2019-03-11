#include <iostream>
#include "cuda_kernel.hpp"
int main(int argc, char * argv[]){
	float *A, *B, *C;
	int m, k, n;
	gemm_kernel(A, B, C, m, k, n);

	return 0;
}
