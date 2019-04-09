#ifndef PARAMETER_STRUCTS
#define PARAMETER_STRUCTS


//SWARM IS ARRAY OF STRUCTURES
typedef struct particle{
	int blocks_per_grid[3];
	int threads_per_block[3];
	int best_block[3];
	int best_thread[3];
	int velocity_thread_x;
	float gemm_time;
	float jacobi_time;
	float total_time;
	float best_time;
}particle_t;
#endif
