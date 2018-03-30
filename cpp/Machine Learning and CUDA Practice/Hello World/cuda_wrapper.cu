#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
//#include "cuda_hello_func.h"

__global__
void helloWorldThreads(int N) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if (idx < N) {
		printf("Hello World! This is thread #%i on block # %i\n", idx, blockIdx.x);
	}
}

void cuda_hello() {
	int numBlocks = 2;
	int threadsPerBlock = 5;
	int maxThreads = 10;
	helloWorldThreads <<<numBlocks, threadsPerBlock >>> (maxThreads);
	cudaDeviceSynchronize();
}