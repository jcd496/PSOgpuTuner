#ifndef PARAMETER_STRUCTS
#define PARAMETER_STRUCTS
#define DIMENSION 2


//SWARM IS ARRAY OF STRUCTURES
typedef struct particle{
	int blocks_per_grid[DIMENSION];
	int threads_per_block[DIMENSION];
	int best_block[DIMENSION];
	int best_thread[DIMENSION];
	int velocity_block[DIMENSION];
	int velocity_thread[DIMENSION];
	float gemm_time;
	float jacobi_time;
	float total_time;
}particle_t;
#endif
