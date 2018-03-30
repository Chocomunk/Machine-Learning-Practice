#include <cuda.h>
#include <cuda_runtime.h>

__global__
void k_VecAdd(int n, float *a, float *b) {
	int idx = threadIdx.x + blockDim.x*blockIdx.x;
	if (idx < n) {
		b[idx] = a[idx] + b[idx];
	}
}
extern void cuda_VecAdd(int n, float *a, float *b, float *c) {
	size_t size = n * sizeof(float);

	float *d_a;
	cudaMalloc(&d_a, size);
	float *d_b;
	cudaMalloc(&d_b, size);

	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

	int threadsPerBlock = 256;
	int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
	k_VecAdd <<<blocksPerGrid, threadsPerBlock >>> (n, d_a, d_b);

	cudaMemcpy(c, d_b, size, cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);
}