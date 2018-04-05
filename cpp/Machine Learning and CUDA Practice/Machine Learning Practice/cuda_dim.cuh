#pragma once
#include "cuda_headers.cuh"

struct c_dim3 : public dim3 {
public:
	__host__ __device__ c_dim3(unsigned int x = 1, unsigned int y = 1, unsigned int z = 1) :dim3(x, y, z) {};
	__host__ __device__ c_dim3::c_dim3(const dim3& a) {x = a.x; y = a.y; z = a.z;}

	__host__ __device__ int c_dim3::prod() const {return x*y*z; }

	const c_dim3 _sum(const dim3& b) const;
	const c_dim3 _mul(const dim3& b) const;
	const c_dim3 _div(const dim3& b) const;
	const c_dim3 _subi(const int b) const;

	__device__ const c_dim3 c_dim3::sum(const dim3& b) const { return c_dim3(x + b.x, y + b.y, z + b.z); }
	__device__ const c_dim3 c_dim3::mul(const dim3& b) const { return c_dim3(x*b.x, y*b.y, z*b.z); }
	__device__ const c_dim3 c_dim3::div(const dim3& b) const { return c_dim3(x / b.x, y / b.y, z / b.z); }
	__device__ const c_dim3 c_dim3::subi(const int b) const { return c_dim3(x - b, y - b, z - b); }

	const c_dim3 operator+(const dim3& b) const;
	const c_dim3 operator/(const dim3& b) const;
	const c_dim3 operator*(const dim3& b) const;

	const c_dim3 operator-(const int b) const;
};

c_dim3 dimMin(const c_dim3& a, const c_dim3& b);

inline __device__ int getGlobalIndex() {
	c_dim3 t_i(threadIdx); c_dim3 b_i(blockIdx); c_dim3 b_d(blockDim); c_dim3 g_d(gridDim);
	c_dim3 totalIdx = t_i.sum(b_i.mul(b_d));
	c_dim3 totalDim = b_d.mul(g_d);
	return totalIdx.x + totalIdx.y * totalDim.x + totalIdx.z * totalDim.y;
}
