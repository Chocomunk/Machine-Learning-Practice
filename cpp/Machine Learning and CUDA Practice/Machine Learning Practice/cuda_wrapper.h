#pragma once
#include "cuda_dim.cuh"

void cuda_MatAdd(const c_dim3& dim, float *a, float *b, float *c);
void cuda_MatDot(const c_dim3& dim, float *a, float *b, float *c);

float* matAdd(const c_dim3& dim, float *a, float *b);
float* matDot(const c_dim3& dim, float* a, float* b);
