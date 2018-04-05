#include "cuda_headers.cuh"
#include "cuda_dim.cuh"

__global__
void k_MatAdd(const c_dim3& dim, float *a, float *b) {
	int idx = getGlobalIndex();
	if (idx < dim.prod()) {
		b[idx] = a[idx] + b[idx];
	}
}

__global__
void k_MatDot(const c_dim3& dim, float *a, float *b) {
	int idx = getGlobalIndex();
	if (idx < dim.prod()) {
		b[idx] = a[idx] * b[idx];
	}
}

extern void cuda_MatAdd(const c_dim3& dim, float *a, float *b, float *c) {
	size_t size = dim.prod() * sizeof(float);

	float *d_a; cudaMalloc(&d_a, size);
	float *d_b; cudaMalloc(&d_b, size);
	c_dim3 *d_dim; cudaMalloc(&d_dim, sizeof(c_dim3));

	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_dim, &dim, sizeof(c_dim3), cudaMemcpyHostToDevice);

	c_dim3 threadsPerBlock = dimMin(dim, c_dim3(256,256,256));
	c_dim3 blocksPerGrid = (dim + threadsPerBlock - 1) / threadsPerBlock;
	k_MatAdd <<<blocksPerGrid, threadsPerBlock >>> (*d_dim, d_a, d_b);

	cudaMemcpy(c, d_b, size, cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);
}

extern void cuda_MatDot(const c_dim3& dim, float *a, float *b, float *c) {
	size_t size = dim.prod() * sizeof(float);

	float *d_a; cudaMalloc(&d_a, size);
	float *d_b; cudaMalloc(&d_b, size);
	c_dim3 *d_dim; cudaMalloc(&d_dim, sizeof(c_dim3));

	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_dim, &dim, sizeof(c_dim3), cudaMemcpyHostToDevice);

	c_dim3 threadsPerBlock = dimMin(dim, c_dim3(256,256,256));
	c_dim3 blocksPerGrid = (dim + threadsPerBlock - 1) / threadsPerBlock;
	k_MatDot <<<blocksPerGrid, threadsPerBlock >>> (*d_dim, d_a, d_b);

	cudaMemcpy(c, d_b, size, cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);
}
