#include "stdafx.h"
#include "CppUnitTest.h"
#include "../Machine Learning Practice/cuda_wrapper.h"
#include "testUtil.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace MachineLearningPracticeTests
{		
	TEST_CLASS(CudaFunctionsTest)
	{
	public:
		
		TEST_METHOD(VecAddTest)
		{
			float a[5] = { 3,1,4,1,5 };
			float b[5] = { 2,1,7,1,8 };
			float sum[5] = { 5,2,11,2,13 };
			float *c = vecAdd(5, a, b);
			float *d = vecAdd(5, a, b);
			Assert::IsTrue(vecIsEqual(5, c, sum));
		}

	};
}