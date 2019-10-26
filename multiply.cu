#include <stdio.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

// CUDA RumTime API
#define TILE_WIDTH 32

void MatrixMultiplyOnHost(float* M, float* N, float* P, int width)
{
	for(int i=0; i<width; ++i)
	{
		for (int j=0; j<width; ++j)
		{
			float sum = 0;
			for(int k=0; k<width; ++k)
			{
				float a = M[i*width+k];
				float b = N[k*width+j];
				sum += a*b;
			}
			P[i*width+j] = sum;
		}
	}

}

__global__ void MatirxMultiplyKernel(const float* devM, const float* devN, float* devP, const int width, const int tile_width)
{
	__shared__ float sM[TILE_WIDTH][TILE_WIDTH];
	__shared__ float sN[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x;		int by = blockIdx.y;
	int tx = threadIdx.x;		int ty = threadIdx.y;

	int col = bx*tile_width+tx;
	int row = by*tile_width+ty;

	//Initialize accumulator to 0
	float pValue = 0;

	//Multiply and add
	for(int m=0; m<width/tile_width;m++)
	{
		sM[ty][tx] = devM[row*width+(m*tile_width+tx)];
		sN[ty][tx] = devN[col+(m*tile_width+ty)*width];
		//each time bring one element from devM and devN into shared memory
		//threads are blocked until all the threads reach this point
		__syncthreads();

		for(int k=0; k<tile_width; k++)
			pValue += sM[ty][k]*sN[k][tx];
		__syncthreads();
	}
	
	//Write value to device memory - each thread has unique index to write to
	devP[row*width+col] = pValue;
}

void MatrixMultiplyOnDevice(const float* hostM, const float* hostN, float* hostP, const int width, const int tile_width)
{
	int sizeInBytes = width*width*sizeof(float);
	float *devM, *devN, *devP;

	//Allocate M and N on devide
	cudaMalloc((void**)&devM, sizeInBytes);
	cudaMalloc((void**)&devN, sizeInBytes);

	//Allocate P
	cudaMalloc((void**)&devP, sizeInBytes);

	//Copy M and N from host to device
	cudaMemcpy(devM, hostM, sizeInBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(devN, hostN, sizeInBytes, cudaMemcpyHostToDevice);

	//Setup thread/block execution configuration
	dim3 dimBlocks(tile_width,tile_width); //Each block has (width,width) threads
	dim3 dimGrid(width/tile_width,width/tile_width); //Launch 1 block


	//Launch the kernel
	clock_t begin = clock();
	MatirxMultiplyKernel<<<dimGrid,dimBlocks>>>(devM,devN,devP,width,tile_width);
	clock_t end = clock();
	float elapsed_secs = float(end - begin) / CLOCKS_PER_SEC;
	printf("Matrix Multiply on Device: %fs\n",elapsed_secs);
	

	//Copy P matrix from device to host
	cudaMemcpy(hostP, devP, sizeInBytes, cudaMemcpyDeviceToHost);

	//Free allocated memory
	cudaFree(devM); cudaFree(devN); cudaFree(devP);
}

void PrintMatrix(float* M, int width)
{
	for(int i=0; i<width; i++)
	{
		for(int j=0; j<width; j++)
		{
			printf("%f ",M[i*width+j]);
		}
		printf("\n");
	}
}

int main()
{
	int width = 1024;
	int tile_width = 32;
	int size = width*width;

	float* M = new float[size];
	float* N = new float[size];
	float* P = new float[size];
	float* Q = new float[size];
	srand (time(NULL));
	for(int i=0; i<size; i++)
	{
		M[i] = rand() / (RAND_MAX + 1.);
		N[i] = rand() / (RAND_MAX + 1.);
	}

	//multiply on host
	clock_t begin = clock();
	MatrixMultiplyOnHost(M,N,P,width);
	clock_t end = clock();
	float elapsed_secs = float(end - begin) / CLOCKS_PER_SEC;
	printf("Matrix Multiply on Host: %fs\n",elapsed_secs);
	//std::cout << "Matrix Multiply on Host: " << elapsed_secs << std::endl;
	
	//multiply on device
	//1. Copy M,N matrices to device
	//2. M*N on device
	//3. Copy P matrix to host and output
	MatrixMultiplyOnDevice(M,N,Q,width,tile_width);

	float avg_err = 0;
	for(int i=0; i<size; i++)
		avg_err += fabs(P[i]-Q[i]);
	avg_err /= size;
	printf("Average error is: %f\n",avg_err);
	//PrintMatrix(M,width);
	//PrintMatrix(N,width);
	//PrintMatrix(P,width);
	//PrintMatrix(Q,width);

	return 0;
}