#include <iostream>
#include <random>
#include <float.h>
#include "cuda_kernel.hpp"
#define WARP 32
#define MAX_THREADS_PER_BLOCK 32//1024
#define MAX_BLOCKS_PER_GRID 32  //65535
#define NUM_PARTICLES 1
#define MAX_ITER 1
#define phi_p 2
#define phi_g 2
using namespace std;

void particle_swarm_optimization(){
	default_random_engine generator;
	uniform_int_distribution<int> grid_distribution(1 , MAX_BLOCKS_PER_GRID);
	uniform_int_distribution<int> block_distribution(1 , MAX_THREADS_PER_BLOCK);
	uniform_int_distribution<int> velocity_grid_dist(-MAX_BLOCKS_PER_GRID, MAX_BLOCKS_PER_GRID);
	uniform_int_distribution<int> velocity_block_dist(-MAX_THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK);
	//SWARM BEST POSITION
	float swarm_best_total_time = FLT_MAX;
	int swarm_best_block[DIMENSION];
	int swarm_best_thread[DIMENSION];

	//INITIALIZE SWARM
	particle_t particles[NUM_PARTICLES];
	for(int i=0; i<NUM_PARTICLES; i++){
		for(int j=0; j<DIMENSION; j++){
			//INITIALIZE POSITION VECTOR
			int block = grid_distribution(generator);
			int thread = block_distribution(generator);
			particles[i].blocks_per_grid[j] = block;
			particles[i].threads_per_block[j] = thread;
			//UPDATE PARTICLES BEST KNOWN POSITION
			particles[i].best_block[j] = block;
			particles[i].best_thread[j] = thread;
			//INITIALIZE VELOCITY
			int Vb = velocity_grid_dist(generator);
			int Vt = velocity_block_dist(generator);
			particles[i].velocity_block[j] = Vb;
			particles[i].velocity_thread[j] = Vt;
		}
		//UPDATE GLOBAL BEST HERE??
		dim3 blocksPerGrid(particles[i].blocks_per_grid[0],particles[i].blocks_per_grid[1]);
		dim3 threadsPerBlock(particles[i].threads_per_block[0],particles[i].threads_per_block[1]);
		kernel_wrapper(i, blocksPerGrid, threadsPerBlock, particles);	
		particles[i].best_time = particles[i].total_time;
	}
	uniform_real_distribution<float> param_dist(0,1);
	int iter=0;
	while(iter<MAX_ITER){
		for(int i=0; i<NUM_PARTICLES; i++){
			for(int j=0; j<DIMENSION; j++){
				float r_p = param_dist(generator);
				float r_g = param_dist(generator);
				particles[i].velocity_block[j] += int(phi_p*r_p*(particles[i].best_block[j]-particles[i].blocks_per_grid[j]) + 
					phi_g*r_g*(swarm_best_block[j]-particles[i].blocks_per_grid[j]));
				particles[i].velocity_thread[j] += int(phi_p*r_p*(particles[i].best_thread[j]-particles[i].threads_per_block[j]) + 
					phi_g*r_g*(swarm_best_thread[j]-particles[i].threads_per_block[j]));


				particles[i].blocks_per_grid[j] += particles[i].velocity_block[j];
				particles[i].threads_per_block[j] += particles[i].velocity_thread[j];

			}
			//2 dimensional. not workable into 3 this way
			dim3 blocksPerGrid(particles[i].blocks_per_grid[0],particles[i].blocks_per_grid[1]);
			dim3 threadsPerBlock(particles[i].threads_per_block[0],particles[i].threads_per_block[1]);
			kernel_wrapper(i, blocksPerGrid, threadsPerBlock, particles);
			if(particles[i].total_time < particles[i].best_time){
				particles[i].best_time = particles[i].total_time;
				for(int j=0; j<DIMENSION; j++){
					particles[i].best_block[j]=particles[i].blocks_per_grid[j];
					particles[i].best_thread[j]=particles[i].threads_per_block[j];
					if(particles[i].best_time < swarm_best_total_time){
						swarm_best_total_time = particles[i].best_time;
						swarm_best_block[j] = particles[i].best_block[j];
						swarm_best_thread[j] = particles[i].best_thread[j];
					}
				}
			}
		}
		printf("best time %f\n", swarm_best_total_time);
		iter++;
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
