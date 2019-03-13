#include <iostream>
#include "cuda_kernel.hpp"
#include "param_struct.hpp"
#define WARP 32
#define MAX_THREADS_PER_BLOCK 1024
int main(int argc, char * argv[]){
	
	//JUST TESTING WITH BELOW
	dim3 blocksPerGrid(3,3);
	dim3 threadsPerBlock(16,16);
	record_t records[5];
	kernel_wrapper(4, blocksPerGrid, threadsPerBlock, records);
	return 0;
}
