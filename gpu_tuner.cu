#include <iostream>
#include <random>
#include <float.h>
#include <math.h>
#include <omp.h>
#include "cuda_kernel.hpp"
#define WARP 32
#define DIMENSION 2
#define PROBLEM_SIZE 1300
#define MAX_THREADS_PER_BLOCK 1024
#define MAX_BLOCKS_PER_GRID 65535
#define NUM_PARTICLES 3
#define MAX_ITER 2
#define phi_p 0.01
#define phi_g 0.01
#define THREADS_PER_DEVICE 1
using namespace std;
int load_position(int id, particle_t * particles, int thread_x, bool *explored_x){
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
	int num_gpus;
	cudaGetDeviceCount(&num_gpus);
	printf("Number of GPUs: %d\n", num_gpus);
	omp_set_num_threads(THREADS_PER_DEVICE*num_gpus);
	//MASTER ARRAY OF BEST PARTICLES
	particle_t best_particles[THREADS_PER_DEVICE*num_gpus];

	#pragma omp parallel shared(best_particles, num_gpus)
	{
		int host_thread = omp_get_thread_num();
		int world_size = omp_get_num_threads();
		if(host_thread == 0)
			printf("number of Host Threads: %d\n", world_size);
		int chunk_size = MAX_THREADS_PER_BLOCK/world_size;
		//ASSIGN DEVICE
		cudaSetDevice(host_thread % num_gpus);
		int device_id;
		cudaGetDevice(&device_id);
		//DISTRIBUTIONS TO EXPLORE PARAMETER SPACE
		default_random_engine generator;
		uniform_int_distribution<int> block_distribution(chunk_size*host_thread , chunk_size*(host_thread+1));
		uniform_int_distribution<int> velocity_block_dist(-2, 2);
		bool explored_x[MAX_THREADS_PER_BLOCK+1] = {00};
		//SWARM BEST POSITION
		float swarm_best_total_time = FLT_MAX;
		int swarm_best_block[DIMENSION];
		int swarm_best_thread[DIMENSION];
		//INITIALIZE SWARM
		particle_t particles[NUM_PARTICLES];
		//LAUNCH DUMMY KERNEL TO REDUCE LATENCY
		dummy_kernel<<<1,1>>>(0);
		cudaDeviceSynchronize();
		for(int i=0; i<NUM_PARTICLES; i++){
			//INITIALIZE POSITION VECTOR
			//if(i==0) load_position(i, particles, 1024);
			//else
				while(load_position(i, particles, block_distribution(generator), explored_x));

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
			
			printf("Parameters: grid %d x %d, block %d x %d From Device: %d\n", particles[i].blocks_per_grid[0], particles[i].blocks_per_grid[1], 
				particles[i].threads_per_block[0], particles[i].threads_per_block[1], device_id);
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

				//POSITION UPDATE, if thread.x chosen by position update is invalid, systematically search for a valid thread.x starting at begining of 
				//host thread's parameter space
				particles[i].threads_per_block[0] += particles[i].velocity_thread_x;
				if(load_position(i, particles, particles[i].threads_per_block[0], explored_x)){
					int trys = 0;
					int thread_x = host_thread*chunk_size;
					while(load_position(i, particles, thread_x, explored_x)){
						//if has tried every thread option in host threads parameter space, break
						if(trys > chunk_size ){
							BREAK = 01;
							printf("Could Not Converge\n");
							break;
						}
						trys++;
						thread_x = host_thread*chunk_size + trys;
					}
				}
				if(BREAK){
					iter=MAX_ITER;
					break;
				}
				//LAUNCH KERNELS TO EVALUATE POSITION VECTOR, not compatible with 3D kernels
				dim3 blocksPerGrid(particles[i].blocks_per_grid[0],particles[i].blocks_per_grid[1]);//, particles[i].blocks_per_grid[2]);
				dim3 threadsPerBlock(particles[i].threads_per_block[0],particles[i].threads_per_block[1]);//, particles[i].threads_per_block[2]);
				kernel_wrapper(i, blocksPerGrid, threadsPerBlock, particles, PROBLEM_SIZE);
					
				printf("Parameters: grid %d x %d, block %d x %d From Device: %d\n", particles[i].blocks_per_grid[0], particles[i].blocks_per_grid[1], 
					particles[i].threads_per_block[0], particles[i].threads_per_block[1], device_id);
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
		//update master record of best particles
		best_particles[host_thread].best_time = swarm_best_total_time;
		for(int j=0; j<DIMENSION; j++){
			best_particles[host_thread].blocks_per_grid[j] = 0;
			best_particles[host_thread].threads_per_block[j] = 0;
			best_particles[host_thread].best_block[j] = swarm_best_block[j];
			best_particles[host_thread].best_thread[j] = swarm_best_thread[j];
		} 
	}
	int best_particle = 0;
	for(int i=1; i<THREADS_PER_DEVICE*num_gpus; i++){
		if(best_particles[i].best_time < best_particles[best_particle].best_time)
			best_particle = i;
	}
	printf("Best Time %f seconds\n", best_particles[best_particle].best_time/1e3);
	printf("Best Parameters: Grid %d x %d, Thread %d x %d\n", best_particles[best_particle].best_block[0], best_particles[best_particle].best_block[1], 
		best_particles[best_particle].best_thread[0], best_particles[best_particle].best_thread[1]);
	//printf("Best Time %f seconds\n", swarm_best_total_time/1e3);
	//printf("Best Parameters: Grid %d x %d, Thread %d x %d\n", swarm_best_block[0], swarm_best_block[1], swarm_best_thread[0], swarm_best_thread[1]);
}


int main(int argc, char * argv[]){
	
	particle_swarm_optimization();
	return 0;
}
