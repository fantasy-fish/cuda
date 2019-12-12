# cuda
cuda programming exercise
1. Naive Matrix Multiplication
	* For a 32*32 matrix multiplication with float numbers, elapsed time on Host is 0.000131s
	* Elapsed time on Device is 0.000019s if run with 32*32 threads
	* Size of matrix limited by the number of threads allowed in a thread block, which is 1024 with CUDA toolkit 10
2. Advanced Matrix Multiplication
	* Split the matrix into tiles, with each tile assigned to a block
	* Each tile can access the shared memory instead of accessing the global memory directly
	* For a 1024*1024 matrix with the tile size of 32*32, it takes 14.500724s on the host, and 0.000022s on the device
	* Below is a nsight profile screenshot
	* ![nsight](https://github.com/fantasy-fish/cuda/blob/master/nsight.png)
3. Flocking Simulation
	* Based on the Reynolds Boids algorithm
	* With two levels of optimization: a uniform grid, and a uniform grid with semi-coherent memory access
	* Below are some results with 1k, 10k and 100k boids(particles)
		* ![boids_1k](https://github.com/fantasy-fish/cuda/blob/master/cuda_flocking/results/1k.gif)
		* ![boids_10k](https://github.com/fantasy-fish/cuda/blob/master/cuda_flocking/results/10k.gif)
		* ![boids_100k](https://github.com/fantasy-fish/cuda/blob/master/cuda_flocking/results/100k.gif)
