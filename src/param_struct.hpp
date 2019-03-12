#ifndef PARAMETER_STRUCTS
#define PARAMETER_STRUCTS


struct parameters{
	dim3 threads_per_block;
	dim3 blocks_per_grid;
};

//STRUCTURE OF ARRAYS
/*struct param_time_record{
	struct parameters * parameters;
	float  * gemm;
};*/
//ARRAY OF STRUCTURES

typedef struct param_time_record{
	struct parameters parameters;
	float gemm;
}record_t;

#endif
