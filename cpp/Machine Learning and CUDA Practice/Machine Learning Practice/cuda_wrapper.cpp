#include "cuda_wrapper.h"

float* matAdd(const c_dim3& dim, float *a, float *b) {
	float *out = new float[dim.prod()];
	cuda_MatAdd(dim, a, b, out);
	return out;
}

float* matDot(const c_dim3& dim, float* a, float* b) {
	float *out = new float[dim.prod()];
	cuda_MatDot(dim, a, b, out);
	return out;
}
