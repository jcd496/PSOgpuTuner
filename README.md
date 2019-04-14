# PSOgpuTuner

a gpu tuner using a distributed particle swarm optimization algorithm to discover best kernel launch parameters for applications involving GEMM and Jacobi Smoothers.  
May be extended to arbitrary applications by supplying a kernel_wrapper() function with gpu kernels.
For efficient extension to other applications, supply one-time memory copy to gpu and one-time gpu-free functions.

void mem_to_device(device_pointer_t * pointers, int problem_size)

void free_device(device_pointer_t * pointers)

For simple extension to other applications, implement cuda memory management in kernel_wrapper().
