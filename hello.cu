#include <cstdio>

__global__ void helloFromGpu(void)
{
	//printf("hello world from GPU!\n");
	printf("hello world from GPU!\n");
}

int main(void)
{
	printf("Hello World from CPU!\n");

	helloFromGpu<<<1,10>>>();
	cudaDeviceReset();

	return 0;
}