#include <iostream>
#include <random>
#include <float.h>
#include <math.h>
#include "cuda_kernel.hpp"
#define WARP 32
#define DIMENSION 2
#define PROBLEM_SIZE 1024
#define MAX_THREADS_PER_BLOCK 1024
#define MAX_BLOCKS_PER_GRID 65535
#define NUM_PARTICLES 5
#define MAX_ITER 5
#define phi_p 1
#define phi_g 1
using namespace std;
int load_position(int id, particle_t * particles, int thread_x){
	if(thread_x < 1 || thread_x >= MAX_THREADS_PER_BLOCK) return 1;
	particles[id].threads_per_block[0] = thread_x;
	//block.x to fit problem size
	particles[id].blocks_per_grid[0] = ceil(float(PROBLEM_SIZE)/float(thread_x));
	//thread.y to fit block size constraints
	particles[id].threads_per_block[1] = MAX_THREADS_PER_BLOCK/thread_x;
	//block.y to fit problem size and block size constraints
	int thread_y = particles[id].threads_per_block[1];
	particles[id].blocks_per_grid[1] = (thread_y) ? ceil(float(PROBLEM_SIZE)/float(thread_y)) : 0;
	//only 2D supported
	particles[id].threads_per_block[2] = 0;
	particles[id].blocks_per_grid[2] = 0;
	return 0;
}
void particle_swarm_optimization(){
	default_random_engine generator;
	uniform_int_distribution<int> block_distribution(1 , 1024);
	uniform_int_distribution<int> velocity_grid_dist(-MAX_BLOCKS_PER_GRID, MAX_BLOCKS_PER_GRID);
	uniform_int_distribution<int> velocity_block_dist(-5, 5);
	//SWARM BEST POSITION
	float swarm_best_total_time = FLT_MAX;
	int swarm_best_block[DIMENSION];
	int swarm_best_thread[DIMENSION];

	//INITIALIZE SWARM
	particle_t particles[NUM_PARTICLES];
	for(int i=0; i<NUM_PARTICLES; i++){
		//INITIALIZE POSITION VECTOR
		while(load_position(i, particles, block_distribution(generator)));
		for(int j=0; j<DIMENSION; j++){
			//UPDATE PARTICLES BEST KNOWN POSITION
			particles[i].best_block[j] = particles[i].blocks_per_grid[j];
			particles[i].best_thread[j] = particles[i].threads_per_block[j];

			//INITIALIZE VELOCITY
			int Vb = velocity_grid_dist(generator);
			int Vt = velocity_block_dist(generator);
			particles[i].velocity_block[j] = Vb;
			particles[i].velocity_thread[j] = Vt;
		}
		//UPDATE PARTICLE BEST AND GLOBAL BEST
		dim3 blocksPerGrid(particles[i].blocks_per_grid[0],particles[i].blocks_per_grid[1]);
		dim3 threadsPerBlock(particles[i].threads_per_block[0],particles[i].threads_per_block[1]);
		kernel_wrapper(i, blocksPerGrid, threadsPerBlock, particles, PROBLEM_SIZE);	
		particles[i].best_time = particles[i].total_time;
		if(particles[i].best_time < swarm_best_total_time){
			swarm_best_total_time = particles[i].best_time;
			for(int j=0; j<DIMENSION; j++){
				swarm_best_block[j] = particles[i].best_block[j];
				swarm_best_thread[j] = particles[i].best_thread[j];
			}

		}
	}
	uniform_real_distribution<float> param_dist(0,1);
	int iter=0;
	while(iter<MAX_ITER){
		for(int i=0; i<NUM_PARTICLES; i++){
			float r_p = param_dist(generator);
			float r_g = param_dist(generator);
			//VELOCITY UPDATE, position vector built around thread.x, so that is the only velocity that is necessary
			//particles[i].velocity_block[j] += int(phi_p*r_p*(particles[i].best_block[j]-particles[i].blocks_per_grid[j]) + 
			//	phi_g*r_g*(swarm_best_block[j]-particles[i].blocks_per_grid[j]));
			particles[i].velocity_thread[0] += int(phi_p*r_p*(particles[i].best_thread[0]-particles[i].threads_per_block[0]) + 
				phi_g*r_g*(swarm_best_thread[0]-particles[i].threads_per_block[0]));
		
			//printf("V %d\n", particles[i].velocity_thread[0]);	
			//POSITION UPDATE, dont allow thread.x < 1
			particles[i].threads_per_block[0] += particles[i].velocity_thread[0];
			if(load_position(i, particles, particles[i].threads_per_block[0]))
				load_position(i, particles, block_distribution(generator));

			//printf("grid %d %d, thread %d %d\n", particles[i].blocks_per_grid[0], particles[i].blocks_per_grid[1], particles[i].threads_per_block[0], particles[i].threads_per_block[1]);

			//LAUNCH KERNELS TO EVALUATE POSITION VECTOR, not compatible with 3D kernels
			dim3 blocksPerGrid(particles[i].blocks_per_grid[0],particles[i].blocks_per_grid[1]);//, particles[i].blocks_per_grid[2]);
			dim3 threadsPerBlock(particles[i].threads_per_block[0],particles[i].threads_per_block[1]);//, particles[i].threads_per_block[2]);
			kernel_wrapper(i, blocksPerGrid, threadsPerBlock, particles, PROBLEM_SIZE);
			//COMMUNICATE TO SWARM
			if(particles[i].total_time < particles[i].best_time){
				particles[i].best_time = particles[i].total_time;
				for(int j=0; j<DIMENSION; j++){
					particles[i].best_block[j]=particles[i].blocks_per_grid[j];
					particles[i].best_thread[j]=particles[i].threads_per_block[j];
				}
				if(particles[i].best_time < swarm_best_total_time){
					swarm_best_total_time = particles[i].best_time;
					for(int j=0; j<DIMENSION; j++){
						swarm_best_block[j] = particles[i].best_block[j];
						swarm_best_thread[j] = particles[i].best_thread[j];
					}
				}
			}
			
		}
		iter++;
	}
	printf("best time %f seconds\n", swarm_best_total_time/1e3);
	printf("Grid %d %d, Thread %d %d\n", swarm_best_block[0], swarm_best_block[1], swarm_best_thread[0], swarm_best_thread[1]);
}


int main(int argc, char * argv[]){
	
	//JUST TESTING WITH BELOW
/*	dim3 blocksPerGrid(147,8);
	dim3 threadsPerBlock(7,146);
	particle_t particles[5];
	kernel_wrapper(4, blocksPerGrid, threadsPerBlock, particles, 1024);
*/	

	particle_swarm_optimization();
	return 0;
}
