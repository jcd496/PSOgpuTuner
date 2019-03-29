#include <iostream>
#include <random>
#include "cuda_kernel.hpp"
#include "particle_struct.hpp"
#define WARP 32
#define MAX_THREADS_PER_BLOCK 32//1024
#define MAX_BLOCKS_PER_GRID 32  //65535
#define NUM_PARTICLES 10
using namespace std;

void particle_swarm_optimization(){
	default_random_engine generator;
	uniform_int_distribution<int> grid_distribution(1 , MAX_BLOCKS_PER_GRID);
	uniform_int_distribution<int> block_distribution(1 , MAX_THREADS_PER_BLOCK);
	uniform_int_distribution<int> velocity_grid_dist(-MAX_BLOCKS_PER_GRID, MAX_BLOCKS_PER_GRID);
	uniform_int_distribution<int> velocity_block_dist(-MAX_THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK);

	particle_t particles[NUM_PARTICLES];
	for(int i=0; i<NUM_PARTICLES; i++){
		for(int j=0; j<DIMENSION; j++){
			//INITIALIZE POSITION VECTOR
			int block = grid_distribution(generator);
			int thread = block_distribution(generator);
			particles[i].blocks_per_grid[j] = block;
			particles[i].threads_per_block[j] = thread;
			particles[i].best_block[j] = block;
			particles[i].best_thread[j] = thread;
			//UPDATE GLOBAL BEST HERE??
			
			//INITIALIZE VELOCITY
			int Vb = velocity_grid_dist(generator);
			int Vt = velocity_block_dist(generator);
			particles[i].velocity_block[j] = Vb;
			particles[i].velocity_thread[j] = Vt;
		}
	}
	for(int i=0; i<NUM_PARTICLES; i++){
		cout<<particles[i].best_block[0]<<endl;
	}

}


int main(int argc, char * argv[]){
	
	//JUST TESTING WITH BELOW
	/*dim3 blocksPerGrid(32,32);
	dim3 threadsPerBlock(32,32);
	particle_t particles[5];
	kernel_wrapper(4, blocksPerGrid, threadsPerBlock, particles);
	*/

	particle_swarm_optimization();
	return 0;
}
