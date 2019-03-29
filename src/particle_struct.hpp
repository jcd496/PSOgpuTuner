#ifndef PARAMETER_STRUCTS
#define PARAMETER_STRUCTS
#define DIMENSION 2


//SWARM IS ARRAY OF STRUCTURES
typedef struct particle{
	int blocks_per_thread[DIMENSION];
	int threads_per_block[DIMENSION];
	float gemm_time;
	float jacobi_time;
	float total_time;
}particle_t;
#endif
