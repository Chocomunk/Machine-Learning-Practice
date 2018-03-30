#ifndef CUDA_WRAPPER_H
#define CUDA_WRAPPER_H

void cuda_VecAdd(int n, float *a, float *b, float *c);

float* vecAdd(const int n, float *a, float *b) {
	float *out = new float[n];
	cuda_VecAdd(n, a, b, out);
	return out;
}

#endif // !CUDA_WRAPPER_H
