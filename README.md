# cuda
cuda programming exercise
1. Helloworld
2. Naive Matrix Multiplication
	* For a 32*32 matrix multiplication with float numbers, elapsed time on Host is 0.000131s
	* Elapsed time on Device is 0.000019s if run with 32*32 threads
	* Size of matrix limited by the number of threads allowed in a thread block, which is 1024 with CUDA toolkit 10
3. Advanced Matrix Multiplication
	* Split the matrix into tiles, with each tile assigned to a block
	* Each tile can access the shared memory instead of accessing the global memory directly
	* For a 1024*1024 matrix with the tile size of 32*32, it takes 14.500724s on the host, and 0.000022s on the device

