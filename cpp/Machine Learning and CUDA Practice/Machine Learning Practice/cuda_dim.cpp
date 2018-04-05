#include "cuda_dim.cuh"

const c_dim3 c_dim3::_sum(const dim3& b) const { return c_dim3(x + b.x, y + b.y, z + b.z); }
const c_dim3 c_dim3::_mul(const dim3& b) const { return c_dim3(x*b.x, y*b.y, z*b.z); }
const c_dim3 c_dim3::_div(const dim3& b) const { return c_dim3(x / b.x, y / b.y, z / b.z); }
const c_dim3 c_dim3::_subi(const int b) const { return c_dim3(x - b, y - b, z - b); }

const c_dim3 c_dim3::operator+(const dim3& b) const { return _sum(b); }
const c_dim3 c_dim3::operator/(const dim3& b) const { return _div(b); }
const c_dim3 c_dim3::operator*(const dim3& b) const { return _mul(b); }

const c_dim3 c_dim3::operator-(const int b) const { return _subi(b); }

c_dim3 dimMin(const c_dim3& a, const c_dim3& b) {
	return c_dim3(__min(a.x, b.x), __min(a.y, b.y), __min(a.z, b.z));
}
