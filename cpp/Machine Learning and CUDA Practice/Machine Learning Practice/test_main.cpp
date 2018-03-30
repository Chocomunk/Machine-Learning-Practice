#include <stdio.h>
#include "cuda_wrapper.h"

int main(int argc, const char* argv[]) {
	float *c = vecAdd(2, new float[2]{ 1,0 }, new float[2]{ 0,1 });
	printf("%f, %f", c[0], c[1]);
}