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
#define NUM_PARTICLES 30
#define MAX_ITER 10
#define phi_p 0.01
#define phi_g 0.01
using namespace std;
bool explored_x[MAX_THREADS_PER_BLOCK+1] = {00};
int load_position(int id, particle_t * particles, int thread_x){
	if(explored_x[thread_x] || thread_x < 1 || thread_x > MAX_THREADS_PER_BLOCK) return 1;
	particles[id].threads_per_block[0] = thread_x;
	explored_x[thread_x] = 01;
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
	uniform_int_distribution<int> block_distribution(1 , MAX_THREADS_PER_BLOCK/2);
	uniform_int_distribution<int> velocity_block_dist(-2, 2);
	//SWARM BEST POSITION
	float swarm_best_total_time = FLT_MAX;
	int swarm_best_block[DIMENSION];
	int swarm_best_thread[DIMENSION];
	//INITIALIZE SWARM
	particle_t particles[NUM_PARTICLES];
	for(int i=0; i<NUM_PARTICLES; i++){
		//INITIALIZE POSITION VECTOR
		if(i==0) load_position(i, particles, 1024);
		else
			while(load_position(i, particles, block_distribution(generator)));

		for(int j=0; j<DIMENSION; j++){
			//UPDATE PARTICLES BEST KNOWN POSITION
			particles[i].best_block[j] = particles[i].blocks_per_grid[j];
			particles[i].best_thread[j] = particles[i].threads_per_block[j];
		}
		//INITIALIZE VELOCITY
		particles[i].velocity_thread_x = velocity_block_dist(generator);
		
		//UPDATE PARTICLE BEST AND GLOBAL BEST
		dim3 blocksPerGrid(particles[i].blocks_per_grid[0],particles[i].blocks_per_grid[1]);
		dim3 threadsPerBlock(particles[i].threads_per_block[0],particles[i].threads_per_block[1]);
		kernel_wrapper(i, blocksPerGrid, threadsPerBlock, particles, PROBLEM_SIZE);	
		particles[i].best_time = particles[i].total_time;
		
		printf("Parameters: grid %d x %d, block %d x %d\n", particles[i].blocks_per_grid[0], particles[i].blocks_per_grid[1], 
			particles[i].threads_per_block[0], particles[i].threads_per_block[1]);
		printf("Time: %f\n", particles[i].total_time/1e3);
		
		if(particles[i].best_time < swarm_best_total_time){
			swarm_best_total_time = particles[i].best_time;
			for(int j=0; j<DIMENSION; j++){
				swarm_best_block[j] = particles[i].best_block[j];
				swarm_best_thread[j] = particles[i].best_thread[j];
			}

		}
	}
	uniform_real_distribution<float> param_dist(0,1);
	bool BREAK = 00;
	int iter=0;
	while(iter<MAX_ITER){
		for(int i=0; i<NUM_PARTICLES; i++){
			float r_p = param_dist(generator);
			float r_g = param_dist(generator);
			
			//VELOCITY UPDATE, position vector built around thread.x, so that is the only velocity that is necessary
			particles[i].velocity_thread_x += int(phi_p*r_p*(particles[i].best_thread[0]-particles[i].threads_per_block[0]) + 
				phi_g*r_g*(swarm_best_thread[0]-particles[i].threads_per_block[0]));
			//POSITION UPDATE
			particles[i].threads_per_block[0] += particles[i].velocity_thread_x;
			int trys = 0;
			//uniform_int_distribution<int> block_distribution(1 , MAX_THREADS_PER_BLOCK/particles[i].blocks_per_grid[0]);
			while(load_position(i, particles, particles[i].threads_per_block[0])){
				if(trys > 1000000 ){
					BREAK = 01;
					printf("could not converge\n");
					break;
				}
				load_position(i, particles, block_distribution(generator));
				trys++;
			}
			if(BREAK){
				iter=MAX_ITER;
				break;
			}
			//LAUNCH KERNELS TO EVALUATE POSITION VECTOR, not compatible with 3D kernels
			dim3 blocksPerGrid(particles[i].blocks_per_grid[0],particles[i].blocks_per_grid[1]);//, particles[i].blocks_per_grid[2]);
			dim3 threadsPerBlock(particles[i].threads_per_block[0],particles[i].threads_per_block[1]);//, particles[i].threads_per_block[2]);
			kernel_wrapper(i, blocksPerGrid, threadsPerBlock, particles, PROBLEM_SIZE);
				
			printf("Parameters: grid %d x %d, block %d x %d\n", particles[i].blocks_per_grid[0], particles[i].blocks_per_grid[1], 
				particles[i].threads_per_block[0], particles[i].threads_per_block[1]);
			printf("Time: %f\n", particles[i].total_time/1e3);
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
	printf("Best Time %f seconds\n", swarm_best_total_time/1e3);
	printf("Best Parameters: Grid %d x %d, Thread %d x %d\n", swarm_best_block[0], swarm_best_block[1], swarm_best_thread[0], swarm_best_thread[1]);
}


int main(int argc, char * argv[]){
	
	particle_swarm_optimization();
	return 0;
}
